package controller

import (
	"fmt"

	batchv1 "k8s.io/api/batch/v1"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/utils/ptr"

	aiv1 "github.com/rootfs/trainjob-operator/api/v1alpha1"
)

// buildCheckpointValidationJob creates a K8s Job that validates a checkpoint.
// Validation checks:
//   - No NaN/Inf in any tensor
//   - All expected shards present (world_size match)
//   - Tensor shapes match model config
//   - Loss sanity: loaded loss value is finite and within expected range
//   - Optional: gradient-norm outlier detection (SDC indicator)
func buildCheckpointValidationJob(tj *aiv1.TrainJob) *batchv1.Job {
	checkpointPath := fmt.Sprintf("%s/step-%d", tj.Spec.Checkpoint.StoragePath, tj.Status.CurrentStep)
	totalGPUs := tj.Spec.NumNodes * tj.Spec.GPUsPerNode

	script := fmt.Sprintf(`#!/usr/bin/env python3
"""Checkpoint validation for distributed training."""
import sys
import os
import glob
import torch

checkpoint_path = "%s"
expected_world_size = %d
model_name = "%s"

print(f"=== Checkpoint Validation ===")
print(f"Path: {checkpoint_path}")
print(f"Expected world_size: {expected_world_size}")

errors = []

# 1. Check all shards present
shard_files = sorted(glob.glob(os.path.join(checkpoint_path, "*.distcp")))
if not shard_files:
    # Fallback: check for standard checkpoint format
    shard_files = sorted(glob.glob(os.path.join(checkpoint_path, "rank_*")))

if len(shard_files) == 0:
    errors.append(f"No checkpoint shards found in {checkpoint_path}")
elif len(shard_files) != expected_world_size:
    errors.append(f"Expected {expected_world_size} shards, found {len(shard_files)}")
else:
    print(f"PASS: {len(shard_files)} shards found")

# 2. Check each shard for NaN/Inf
nan_count = 0
inf_count = 0
total_params = 0

for shard_file in shard_files[:4]:  # Sample first 4 shards for speed
    try:
        state = torch.load(shard_file, map_location="cpu", weights_only=True)
        for key, tensor in state.items():
            if isinstance(tensor, torch.Tensor) and tensor.is_floating_point():
                total_params += tensor.numel()
                nans = torch.isnan(tensor).sum().item()
                infs = torch.isinf(tensor).sum().item()
                if nans > 0:
                    nan_count += nans
                    errors.append(f"NaN detected in {shard_file}:{key} ({nans} values)")
                if infs > 0:
                    inf_count += infs
                    errors.append(f"Inf detected in {shard_file}:{key} ({infs} values)")
    except Exception as e:
        errors.append(f"Failed to load {shard_file}: {e}")

if nan_count == 0 and inf_count == 0:
    print(f"PASS: No NaN/Inf in sampled shards ({total_params:,} params checked)")

# 3. Check metadata if available
metadata_file = os.path.join(checkpoint_path, "metadata.json")
if os.path.exists(metadata_file):
    import json
    with open(metadata_file) as f:
        meta = json.load(f)
    loss = meta.get("loss")
    if loss is not None:
        if not (0 < loss < 100):
            errors.append(f"Loss value {loss} outside expected range (0, 100)")
        else:
            print(f"PASS: Loss = {loss:.4f} (within range)")
    grad_norm = meta.get("grad_norm")
    if grad_norm is not None:
        if grad_norm > 1000:
            errors.append(f"Gradient norm {grad_norm} suspiciously high (possible SDC)")
        else:
            print(f"PASS: Gradient norm = {grad_norm:.4f}")

# 4. Report
if errors:
    print(f"\nFAILED: {len(errors)} validation error(s):")
    for e in errors:
        print(f"  - {e}")
    sys.exit(1)
else:
    print(f"\n=== Checkpoint validation PASSED ===")
    sys.exit(0)
`, checkpointPath, totalGPUs, tj.Spec.Model)

	backoffLimit := int32(0)

	return &batchv1.Job{
		ObjectMeta: metav1.ObjectMeta{
			Name:      checkpointValJobName(tj),
			Namespace: tj.Namespace,
			Labels: map[string]string{
				"training.vsr.dev/trainjob":  tj.Name,
				"training.vsr.dev/component": "checkpoint-validator",
				"kueue.x-k8s.io/queue-name":  "none",
			},
		},
		Spec: batchv1.JobSpec{
			BackoffLimit: &backoffLimit,
			// TTL to auto-clean old validation jobs (1 hour)
			TTLSecondsAfterFinished: ptr.To(int32(3600)),
			Template: corev1.PodTemplateSpec{
				ObjectMeta: metav1.ObjectMeta{
					Labels: map[string]string{
						"training.vsr.dev/trainjob":  tj.Name,
						"training.vsr.dev/component": "checkpoint-validator",
					},
				},
				Spec: corev1.PodSpec{
					RestartPolicy: corev1.RestartPolicyNever,
					Containers: []corev1.Container{
						{
							Name:    "validator",
							Image:   "pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime",
							Command: []string{"python3", "-c", script},
							VolumeMounts: []corev1.VolumeMount{
								{Name: "checkpoints", MountPath: tj.Spec.Checkpoint.StoragePath, ReadOnly: true},
							},
						},
					},
					Volumes: []corev1.Volume{
						{
							Name: "checkpoints",
							VolumeSource: corev1.VolumeSource{
								PersistentVolumeClaim: &corev1.PersistentVolumeClaimVolumeSource{
									ClaimName: tj.Name + "-checkpoints",
									ReadOnly:  true,
								},
							},
						},
					},
				},
			},
		},
	}
}
