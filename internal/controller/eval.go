package controller

import (
	"fmt"

	batchv1 "k8s.io/api/batch/v1"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

	aiv1 "github.com/rootfs/trainjob-operator/api/v1alpha1"
)

func evalJobName(tj *aiv1.TrainJob) string {
	return fmt.Sprintf("%s-eval-%d", tj.Name, tj.Status.CurrentStep)
}

// buildEvalJob creates a Kubernetes Job that runs post-training evaluation.
// The eval container reads the latest checkpoint and runs benchmarks, writing
// structured results to a JSON file. The reconciler (or a sidecar) reads these
// results and patches status.eval.
func buildEvalJob(tj *aiv1.TrainJob) *batchv1.Job {
	ec := tj.Spec.EvalConfig
	if ec == nil {
		return nil
	}

	image := ec.Image
	if image == "" {
		image = tj.Spec.Image
	}

	gpus := tj.Spec.GPUsPerNode
	if ec.GPUsPerNode != nil {
		gpus = *ec.GPUsPerNode
	}

	labels := map[string]string{
		"training.vsr.dev/trainjob":  tj.Name,
		"training.vsr.dev/component": "eval",
		"kueue.x-k8s.io/queue-name":  "none",
	}

	envVars := []corev1.EnvVar{
		{Name: "CHECKPOINT_DIR", Value: tj.Spec.Checkpoint.StoragePath},
		{Name: "TRAINJOB_NAME", Value: tj.Name},
		{Name: "TRAINJOB_NAMESPACE", Value: tj.Namespace},
		{Name: "EVAL_OUTPUT_FILE", Value: "/var/run/eval/results.json"},
	}

	if ec.DatasetPath != "" {
		envVars = append(envVars, corev1.EnvVar{Name: "EVAL_DATASET_PATH", Value: ec.DatasetPath})
	}
	if ec.PreviousModelPath != "" {
		envVars = append(envVars, corev1.EnvVar{Name: "PREVIOUS_MODEL_PATH", Value: ec.PreviousModelPath})
	}
	if tj.Status.LastCheckpoint != "" {
		envVars = append(envVars, corev1.EnvVar{Name: "CHECKPOINT_PATH", Value: tj.Status.LastCheckpoint})
	}

	backoff := int32(0)
	one := int32(1)

	return &batchv1.Job{
		ObjectMeta: metav1.ObjectMeta{
			Name:      evalJobName(tj),
			Namespace: tj.Namespace,
			Labels:    labels,
		},
		Spec: batchv1.JobSpec{
			BackoffLimit: &backoff,
			Completions:  &one,
			Parallelism:  &one,
			Template: corev1.PodTemplateSpec{
				ObjectMeta: metav1.ObjectMeta{
					Labels: labels,
				},
				Spec: corev1.PodSpec{
					RestartPolicy: corev1.RestartPolicyNever,
					Containers: []corev1.Container{
						{
							Name:    "eval",
							Image:   image,
							Command: ec.Command,
							Args:    ec.Args,
							Env:     envVars,
							Resources: corev1.ResourceRequirements{
								Requests: corev1.ResourceList{
									corev1.ResourceCPU:    resource.MustParse("8"),
									corev1.ResourceMemory: resource.MustParse("64Gi"),
									"nvidia.com/gpu":      resource.MustParse(fmt.Sprintf("%d", gpus)),
								},
								Limits: corev1.ResourceList{
									"nvidia.com/gpu": resource.MustParse(fmt.Sprintf("%d", gpus)),
								},
							},
							VolumeMounts: []corev1.VolumeMount{
								{Name: "checkpoints", MountPath: tj.Spec.Checkpoint.StoragePath},
								{Name: "eval-output", MountPath: "/var/run/eval"},
							},
						},
					},
					Volumes: []corev1.Volume{
						{
							Name: "checkpoints",
							VolumeSource: corev1.VolumeSource{
								PersistentVolumeClaim: &corev1.PersistentVolumeClaimVolumeSource{
									ClaimName: tj.Name + "-checkpoints",
								},
							},
						},
						{
							Name: "eval-output",
							VolumeSource: corev1.VolumeSource{
								EmptyDir: &corev1.EmptyDirVolumeSource{},
							},
						},
					},
				},
			},
		},
	}
}
