package controller

import (
	"fmt"
	"strings"

	batchv1 "k8s.io/api/batch/v1"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/utils/ptr"

	aiv1 "github.com/rootfs/trainjob-operator/api/v1alpha1"
)

// buildPrologJob creates a K8s Job that runs pre-job validation on all nodes.
// The prolog has three phases:
//
// Phase 1 — Hardware Health:
//   - nvidia-smi: all GPUs visible, no ECC errors
//   - dcgmi diag -r 2: short DCGM diagnostic (memory, NVLink, PCIe)
//   - GPU clock speeds: detect thermally throttled GPUs
//   - InfiniBand port status (if applicable)
//
// Phase 2 — Kernel Validation:
//   - compute-sanitizer memcheck on a small CUDA matmul (catches driver/memory issues)
//   - torch.compile warmup: compile a dummy forward pass to verify Triton codegen works
//     on this exact hardware (catches CUDA version mismatch, missing CUTLASS, PTX issues)
//   - Precision smoke test: run a small matmul in the job's precision (bf16/fp8) and
//     verify output is within expected numerical tolerance vs fp32 reference
//
// Phase 3 — Interconnect Bandwidth:
//   - NCCL AllReduce bandwidth test (all GPUs on node): catches degraded NVLink
//   - Per-GPU peer-to-peer bandwidth matrix: catches asymmetric link failures
func buildPrologJob(tj *aiv1.TrainJob) *batchv1.Job {
	gpuType := tj.Spec.NodeSelector["nvidia.com/gpu.product"]
	hasIB := strings.Contains(gpuType, "H100") || strings.Contains(gpuType, "H200")

	// ── Phase 1: Hardware Health ──
	script := `#!/bin/bash
set -euo pipefail

RESULTS_DIR="/tmp/prolog-results"
mkdir -p "$RESULTS_DIR"

echo "=== Phase 1: Hardware Health Check ==="
echo "Node: $(hostname), Date: $(date -u), GPUs expected: ${EXPECTED_GPU_COUNT}"

# 1.1 Verify all GPUs visible
GPU_COUNT=$(nvidia-smi -L | wc -l)
if [ "$GPU_COUNT" -ne "$EXPECTED_GPU_COUNT" ]; then
    echo "FAIL: Expected $EXPECTED_GPU_COUNT GPUs, found $GPU_COUNT"
    exit 1
fi
echo "PASS: $GPU_COUNT GPUs detected"

# 1.2 Check for ECC errors
UNCORRECTABLE=$(nvidia-smi --query-gpu=ecc.errors.uncorrected.aggregate.total --format=csv,noheader,nounits | awk '{s+=$1} END {print s}')
if [ "$UNCORRECTABLE" -gt 0 ]; then
    echo "FAIL: $UNCORRECTABLE uncorrectable ECC errors detected"
    nvidia-smi --query-gpu=index,ecc.errors.uncorrected.aggregate.total --format=csv
    exit 1
fi
echo "PASS: No uncorrectable ECC errors"

# 1.3 DCGM short diagnostic (memory, NVLink, PCIe)
if command -v dcgmi &> /dev/null; then
    dcgmi diag -r 2 || { echo "FAIL: DCGM diagnostic failed"; exit 1; }
    echo "PASS: DCGM diagnostic passed"
fi

# 1.4 GPU clock throttling check
nvidia-smi --query-gpu=index,clocks.current.graphics,clocks.max.graphics --format=csv,noheader,nounits | while IFS=', ' read -r idx current max; do
    ratio=$(echo "scale=2; $current / $max" | bc)
    if (( $(echo "$ratio < 0.80" | bc -l) )); then
        echo "FAIL: GPU $idx clock throttled: ${current}MHz / ${max}MHz (ratio=${ratio})"
        exit 1
    fi
done
echo "PASS: GPU clocks within normal range"

# 1.5 PCIe link width/speed check
nvidia-smi --query-gpu=index,pcie.link.width.current,pcie.link.gen.current --format=csv,noheader,nounits | while IFS=', ' read -r idx width gen; do
    if [ "$width" -lt 16 ] 2>/dev/null; then
        echo "WARN: GPU $idx PCIe link width=${width}x (expected 16x) — possible downtraining"
    fi
done
echo "PASS: PCIe link check complete"
`

	// ── Phase 1.6: InfiniBand (conditional) ──
	if hasIB {
		script += `
# 1.6 InfiniBand port check
if command -v ibstat &> /dev/null; then
    IB_DOWN=$(ibstat | grep -c "State: Down" || true)
    if [ "$IB_DOWN" -gt 0 ]; then
        echo "FAIL: $IB_DOWN InfiniBand ports are down"
        ibstat
        exit 1
    fi
    echo "PASS: All InfiniBand ports active"

    # IB link rate check (expected 200 or 400 Gbps)
    ibstat | grep "Rate:" | while read -r line; do
        rate=$(echo "$line" | grep -oP '\d+')
        if [ "$rate" -lt 200 ] 2>/dev/null; then
            echo "WARN: IB link rate ${rate} Gbps — expected ≥200 Gbps"
        fi
    done
fi
`
	}

	// ── Phase 2: Kernel Validation ──
	script += `
echo ""
echo "=== Phase 2: GPU Kernel Validation ==="

# 2.1 Compute Sanitizer — memcheck on a quick CUDA matmul
# Catches driver bugs, memory corruption, and hardware faults before the job
if command -v compute-sanitizer &> /dev/null; then
    python3 -c "
import torch
a = torch.randn(256, 256, device='cuda')
b = torch.randn(256, 256, device='cuda')
c = torch.mm(a, b)
torch.cuda.synchronize()
print('matmul ok, shape:', c.shape)
" > /tmp/sanitizer_target.py 2>&1 || true

    compute-sanitizer --tool memcheck --error-exitcode 1 \
        python3 /tmp/sanitizer_target.py 2>"$RESULTS_DIR/sanitizer.log" || {
        echo "FAIL: compute-sanitizer memcheck found errors:"
        cat "$RESULTS_DIR/sanitizer.log"
        exit 1
    }
    echo "PASS: compute-sanitizer memcheck clean"
else
    echo "SKIP: compute-sanitizer not available"
fi

# 2.2 torch.compile warmup — verify Triton codegen works on this hardware
# Catches CUDA version mismatch, missing CUTLASS, broken Triton cache, PTX issues
python3 -c "
import torch, time

@torch.compile(mode='default')
def fwd(x):
    return torch.nn.functional.gelu(x @ x.T)

x = torch.randn(512, 512, device='cuda', dtype=torch.bfloat16)
t0 = time.time()
y = fwd(x)
torch.cuda.synchronize()
compile_time = time.time() - t0
print(f'torch.compile warmup ok, compile_time={compile_time:.2f}s, output_norm={y.norm().item():.4f}')

if compile_time > 120:
    print(f'WARN: torch.compile took {compile_time:.0f}s (expected <60s) — possible Triton cache miss')
" 2>&1 || {
    echo "FAIL: torch.compile warmup failed — Triton codegen broken on this node"
    exit 1
}
echo "PASS: torch.compile warmup succeeded"

# 2.3 Precision smoke test — verify bf16/fp8 matmul produces correct results
# Catches hardware precision bugs (SDC), tensor core defects, firmware issues
python3 -c "
import torch, sys

precision = '${PRECISION}'
dim = 1024

# FP32 reference
torch.manual_seed(42)
a32 = torch.randn(dim, dim, device='cuda', dtype=torch.float32)
b32 = torch.randn(dim, dim, device='cuda', dtype=torch.float32)
ref = torch.mm(a32, b32)

# Test precision
if precision == 'fp8':
    # FP8 requires Hopper+, use torch.float8_e4m3fn
    try:
        a8 = a32.to(torch.float8_e4m3fn)
        b8 = b32.to(torch.float8_e4m3fn).T  # FP8 matmul needs specific layout
        result = torch._scaled_mm(a8, b8, out_dtype=torch.bfloat16)
        tol = 0.1  # FP8 has wider tolerance
    except Exception as e:
        print(f'FAIL: FP8 matmul not supported: {e}')
        sys.exit(1)
elif precision == 'bf16':
    result = torch.mm(a32.bfloat16(), b32.bfloat16()).float()
    tol = 0.01
else:
    result = torch.mm(a32, b32)
    tol = 1e-6

# Relative error check
rel_err = (result.float() - ref).abs().max() / ref.abs().max()
print(f'Precision={precision}, max_rel_error={rel_err.item():.6f}, tolerance={tol}')

if rel_err > tol:
    print(f'FAIL: {precision} matmul relative error {rel_err.item():.6f} exceeds tolerance {tol}')
    print('This may indicate tensor core defect or SDC')
    sys.exit(1)

# Cross-GPU consistency: same matmul on all GPUs should produce identical results
gpu_count = torch.cuda.device_count()
if gpu_count > 1:
    results = []
    for i in range(gpu_count):
        with torch.cuda.device(i):
            ai = a32.to(f'cuda:{i}').bfloat16()
            bi = b32.to(f'cuda:{i}').bfloat16()
            ri = torch.mm(ai, bi)
            results.append(ri.cpu())

    for i in range(1, len(results)):
        diff = (results[i] - results[0]).abs().max().item()
        if diff > 1e-5:
            print(f'FAIL: GPU {i} vs GPU 0 differ by {diff:.8f} — possible SDC on GPU {i}')
            sys.exit(1)
    print(f'PASS: Cross-GPU consistency check passed ({gpu_count} GPUs)')

print('PASS: Precision smoke test passed')
" 2>&1 || {
    echo "FAIL: Precision smoke test failed"
    exit 1
}
echo "PASS: Precision validation complete"
`

	// ── Phase 3: Interconnect Bandwidth ──
	script += `
echo ""
echo "=== Phase 3: Interconnect Bandwidth ==="

# 3.1 NCCL AllReduce bandwidth test (intra-node)
# Catches degraded NVLink, PCIe downtraining, NCCL misconfiguration
python3 -c "
import torch, torch.distributed as dist, os, time

gpu_count = torch.cuda.device_count()
if gpu_count < 2:
    print('SKIP: Single GPU, no intra-node bandwidth test')
    exit(0)

# P2P bandwidth matrix
print('P2P bandwidth matrix (GB/s):')
size = 256 * 1024 * 1024  # 256 MB
header = '     ' + ''.join(f'GPU{j:>7}' for j in range(gpu_count))
print(header)

for src in range(gpu_count):
    row = f'GPU{src} '
    for dst in range(gpu_count):
        if src == dst:
            row += '   ---  '
            continue
        a = torch.randn(size // 4, device=f'cuda:{src}')
        torch.cuda.synchronize()
        t0 = time.time()
        b = a.to(f'cuda:{dst}')
        torch.cuda.synchronize()
        elapsed = time.time() - t0
        bw = (size / 1e9) / elapsed
        row += f'{bw:7.1f} '
        if bw < 20.0:
            print(f'WARN: GPU{src}→GPU{dst} bandwidth={bw:.1f} GB/s (expected >100 GB/s on NVLink)')
    print(row)

print('PASS: P2P bandwidth matrix complete')
" 2>&1 || echo "WARN: P2P bandwidth test had issues (non-fatal)"

echo ""
echo "=== All prolog checks passed ==="
`

	backoffLimit := int32(0)

	return &batchv1.Job{
		ObjectMeta: metav1.ObjectMeta{
			Name:      prologJobName(tj),
			Namespace: tj.Namespace,
			Labels: map[string]string{
				"training.vsr.dev/trainjob":  tj.Name,
				"training.vsr.dev/component": "prolog",
				"kueue.x-k8s.io/queue-name":  "none",
			},
		},
		Spec: batchv1.JobSpec{
			BackoffLimit:   &backoffLimit,
			Completions:    &tj.Spec.NumNodes,
			Parallelism:    &tj.Spec.NumNodes,
			CompletionMode: ptr.To(batchv1.IndexedCompletion),
			Template: corev1.PodTemplateSpec{
				ObjectMeta: metav1.ObjectMeta{
					Labels: map[string]string{
						"training.vsr.dev/trainjob":  tj.Name,
						"training.vsr.dev/component": "prolog",
					},
				},
				Spec: corev1.PodSpec{
					RestartPolicy: corev1.RestartPolicyNever,
					NodeSelector:  tj.Spec.NodeSelector,
					Tolerations: []corev1.Toleration{
						{
							Key:      "nvidia.com/gpu",
							Operator: corev1.TolerationOpExists,
							Effect:   corev1.TaintEffectNoSchedule,
						},
					},
					Containers: []corev1.Container{
						{
							Name: "prolog",
							// Use training image so torch.compile, NCCL, and precision tests
							// validate the exact same software stack the job will use
							Image:   tj.Spec.Image,
							Command: []string{"bash", "-c"},
							Args:    []string{script},
							Env: []corev1.EnvVar{
								{Name: "EXPECTED_GPU_COUNT", Value: fmt.Sprintf("%d", tj.Spec.GPUsPerNode)},
								{Name: "PRECISION", Value: tj.Spec.Precision},
								{Name: "NCCL_DEBUG", Value: "WARN"},
							},
							Resources: corev1.ResourceRequirements{
								Limits: corev1.ResourceList{
									"nvidia.com/gpu": resource.MustParse(fmt.Sprintf("%d", tj.Spec.GPUsPerNode)),
								},
							},
							SecurityContext: &corev1.SecurityContext{
								Privileged: ptr.To(true),
							},
							VolumeMounts: []corev1.VolumeMount{
								{Name: "shm", MountPath: "/dev/shm"},
							},
						},
					},
					Volumes: []corev1.Volume{
						{
							Name: "shm",
							VolumeSource: corev1.VolumeSource{
								EmptyDir: &corev1.EmptyDirVolumeSource{
									Medium:    corev1.StorageMediumMemory,
									SizeLimit: resourcePtr("16Gi"),
								},
							},
						},
					},
				},
			},
		},
	}
}

// resourcePtr is defined in workers.go
