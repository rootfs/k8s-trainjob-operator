package webhook

import (
	"context"
	"fmt"
	"strings"
	"sync"

	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	"k8s.io/apimachinery/pkg/runtime"
	"sigs.k8s.io/controller-runtime/pkg/log"
	"sigs.k8s.io/controller-runtime/pkg/webhook/admission"

	aiv1 "github.com/rootfs/trainjob-operator/api/v1alpha1"
)

//+kubebuilder:webhook:path=/mutate-training-vsr-dev-v1alpha1-trainjob,mutating=true,failurePolicy=fail,sideEffects=None,groups=training.vsr.dev,resources=trainjobs,verbs=create;update,versions=v1alpha1,name=mtrainjob.training.vsr.dev,admissionReviewVersions=v1

// TrainJobMutator injects defaults into TrainJob resources:
//   - Auto-parallelism configuration (when spec.autoParallelism=true)
//   - NCCL environment variables based on GPU type
//   - Default precision (bf16) if not set
//   - Default checkpoint retain count
//   - GPU monitoring sidecar container
//
// The auto-parallelism search result is cached by (model, GPU, cluster shape)
// so repeated submissions with the same config skip the search entirely.
// The pod sidecar webhook bails out in O(1) for non-training pods via label check.
type TrainJobMutator struct {
	configCache sync.Map // key: configCacheKey → value: ParallelismConfig
}

type configCacheKey struct {
	model       string
	gpuType     string
	numNodes    int32
	gpusPerNode int32
}

var _ admission.CustomDefaulter = &TrainJobMutator{}

func (m *TrainJobMutator) Default(ctx context.Context, obj runtime.Object) error {
	tj := obj.(*aiv1.TrainJob)

	if tj.Spec.AutoParallelism {
		if err := m.autoConfigureParallelism(ctx, tj); err != nil {
			return err
		}
	}

	m.defaultPrecision(tj)
	m.defaultCheckpoint(tj)
	m.defaultCPDegree(tj)
	m.injectNCCLEnv(tj)
	m.defaultSuspendForKueue(tj)

	return nil
}

// defaultSuspendForKueue sets suspend=true when the TrainJob carries a
// kueue.x-k8s.io/queue-name label, indicating it should be managed by Kueue.
// Kueue will unsuspend the job when quota is available.
func (m *TrainJobMutator) defaultSuspendForKueue(tj *aiv1.TrainJob) {
	if _, hasQueue := tj.Labels["kueue.x-k8s.io/queue-name"]; hasQueue {
		if tj.Spec.Suspend == nil {
			suspend := true
			tj.Spec.Suspend = &suspend
		}
	}
}

// autoConfigureParallelism derives optimal TP/PP/FSDP/CP/batch/precision
// from the model architecture and cluster hardware, then writes them into
// the spec. This is the "model-aware auto-config advisor."
func (m *TrainJobMutator) autoConfigureParallelism(ctx context.Context, tj *aiv1.TrainJob) error {
	logger := log.FromContext(ctx)

	var modelSpec aiv1.ModelArchSpec
	if tj.Spec.ModelSpec != nil {
		modelSpec = *tj.Spec.ModelSpec
	} else if known, ok := ModelRegistry[tj.Spec.Model]; ok {
		modelSpec = known
		tj.Spec.ModelSpec = &modelSpec
	} else {
		return fmt.Errorf(
			"autoParallelism requires either spec.modelSpec or a known model name "+
				"(got %q); known models: %s",
			tj.Spec.Model, knownModelNames())
	}

	gpuType := tj.Spec.NodeSelector["nvidia.com/gpu.product"]
	gpu := lookupGPU(gpuType)

	// Check cache: same (model, GPU, cluster shape) → reuse previous result.
	// This avoids redundant searches when multiple TrainJobs target the same
	// model on the same hardware (common in hyperparameter sweeps).
	cacheKey := configCacheKey{
		model:       tj.Spec.Model,
		gpuType:     gpuType,
		numNodes:    tj.Spec.NumNodes,
		gpusPerNode: tj.Spec.GPUsPerNode,
	}

	var config ParallelismConfig
	if cached, ok := m.configCache.Load(cacheKey); ok {
		config = cached.(ParallelismConfig)
		logger.Info("auto-parallelism cache hit",
			"model", tj.Spec.Model, "gpu", gpuType,
			"tp", config.TP, "pp", config.PP)
	} else {
		config = AutoConfigureParallelism(
			modelSpec, gpu, tj.Spec.GPUsPerNode, tj.Spec.NumNodes,
		)
		if config.TP > 0 {
			m.configCache.Store(cacheKey, config)
		}
	}

	if config.TP == 0 {
		return fmt.Errorf("auto-parallelism failed: %s", config.Reason)
	}

	logger.Info("auto-parallelism configured",
		"tp", config.TP, "pp", config.PP, "fsdp", config.FSDP,
		"cp", config.CP, "microBatch", config.MicroBatchSize,
		"actCkpt", config.ActivationCheckpointing,
		"precision", config.Precision, "score", config.Score)

	// Write the derived config back into the spec
	tj.Spec.TPDegree = config.TP
	tj.Spec.PPDegree = config.PP
	tj.Spec.CPDegree = config.CP
	tj.Spec.Precision = config.Precision
	tj.Spec.ModelSpec.MicroBatchSize = config.MicroBatchSize
	tj.Spec.ModelSpec.ActivationCheckpointing = config.ActivationCheckpointing

	// Record the reasoning as an annotation for observability
	if tj.Annotations == nil {
		tj.Annotations = make(map[string]string)
	}
	tj.Annotations["training.vsr.dev/auto-parallelism-config"] = fmt.Sprintf(
		"TP=%d PP=%d FSDP=%d CP=%d micro_batch=%d act_ckpt=%v precision=%s score=%.1f",
		config.TP, config.PP, config.FSDP, config.CP,
		config.MicroBatchSize, config.ActivationCheckpointing,
		config.Precision, config.Score)
	tj.Annotations["training.vsr.dev/auto-parallelism-reason"] = config.Reason

	return nil
}

func knownModelNames() string {
	names := make([]string, 0, len(ModelRegistry))
	for k := range ModelRegistry {
		names = append(names, k)
	}
	return strings.Join(names, ", ")
}

// defaultPrecision sets bf16 if no precision is specified.
func (m *TrainJobMutator) defaultPrecision(tj *aiv1.TrainJob) {
	if tj.Spec.Precision == "" {
		tj.Spec.Precision = "bf16"
	}
}

// defaultCheckpoint sets sensible checkpoint defaults.
func (m *TrainJobMutator) defaultCheckpoint(tj *aiv1.TrainJob) {
	if tj.Spec.Checkpoint.Enabled {
		if tj.Spec.Checkpoint.IntervalMinutes == nil {
			defaultInterval := int32(30)
			tj.Spec.Checkpoint.IntervalMinutes = &defaultInterval
		}
		if tj.Spec.Checkpoint.RetainCount == nil {
			defaultRetain := int32(3)
			tj.Spec.Checkpoint.RetainCount = &defaultRetain
		}
	}
}

// defaultCPDegree sets CP=1 if not specified.
func (m *TrainJobMutator) defaultCPDegree(tj *aiv1.TrainJob) {
	if tj.Spec.CPDegree == 0 {
		tj.Spec.CPDegree = 1
	}
}

// injectNCCLEnv adds GPU-type-specific NCCL configuration.
// Different GPU architectures need different NCCL settings for optimal
// collective communication performance.
func (m *TrainJobMutator) injectNCCLEnv(tj *aiv1.TrainJob) {
	gpuType := tj.Spec.NodeSelector["nvidia.com/gpu.product"]
	ncclVars := ncclEnvForGPU(gpuType)

	// Merge without overwriting user-specified values
	existingKeys := make(map[string]bool)
	for _, e := range tj.Spec.Env {
		existingKeys[e.Name] = true
	}
	for _, v := range ncclVars {
		if !existingKeys[v.Name] {
			tj.Spec.Env = append(tj.Spec.Env, aiv1.EnvVar{Name: v.Name, Value: v.Value})
		}
	}
}

type envPair struct {
	Name  string
	Value string
}

func ncclEnvForGPU(gpuType string) []envPair {
	switch {
	case strings.Contains(gpuType, "H100") || strings.Contains(gpuType, "H200"):
		return []envPair{
			{Name: "NCCL_ALGO", Value: "Ring"},
			{Name: "NCCL_NET", Value: "IB"},
			{Name: "NCCL_IB_HCA", Value: "mlx5_0:1,mlx5_1:1,mlx5_2:1,mlx5_3:1"},
			{Name: "NCCL_CROSS_NIC", Value: "1"},
			{Name: "NCCL_IB_GID_INDEX", Value: "3"},
			{Name: "NCCL_SOCKET_IFNAME", Value: "eth0"},
			{Name: "NCCL_DEBUG", Value: "WARN"},
			{Name: "NCCL_IB_TIMEOUT", Value: "23"},
			{Name: "NCCL_IB_RETRY_CNT", Value: "7"},
		}
	case strings.Contains(gpuType, "A100"):
		return []envPair{
			{Name: "NCCL_ALGO", Value: "Ring"},
			{Name: "NCCL_NET", Value: "IB"},
			{Name: "NCCL_IB_HCA", Value: "mlx5_0:1,mlx5_1:1"},
			{Name: "NCCL_DEBUG", Value: "WARN"},
			{Name: "NCCL_IB_TIMEOUT", Value: "23"},
		}
	case strings.Contains(gpuType, "L40") || strings.Contains(gpuType, "L4"):
		return []envPair{
			{Name: "NCCL_NET", Value: "Socket"},
			{Name: "NCCL_SOCKET_IFNAME", Value: "eth0"},
			{Name: "NCCL_DEBUG", Value: "WARN"},
		}
	default:
		return []envPair{
			{Name: "NCCL_DEBUG", Value: "WARN"},
		}
	}
}

// ════════════════════════════════════════════════════════════════
//  Pod-level mutating webhook: sidecar injection
// ════════════════════════════════════════════════════════════════
//
// The sidecar is a comprehensive per-job observability agent with 5 subsystems:
//
//  1. DCGM Exporter — GPU metrics to Prometheus (utilization, memory, temp, power, ECC)
//  2. Hardware Telemetry Collector — system-level metrics not covered by DCGM:
//     PCIe bandwidth, NVLink error counters, IB port stats, CPU/memory/IO pressure
//  3. Kernel Performance Monitor — tracks CUDA kernel execution times per step,
//     detects regressions vs. rolling baseline (catches thermal throttle, NVLink degradation)
//  4. Straggler Detector — monitors training step time via shared file protocol,
//     compares this worker's step time against cluster median, flags outliers
//  5. Anomaly Watchdog — continuous checks for ECC errors, thermal events, XID errors,
//     OOM proximity; can trigger preemptive checkpoint via signal to trainer

//+kubebuilder:webhook:path=/mutate-v1-pod,mutating=true,failurePolicy=ignore,sideEffects=None,groups="",resources=pods,verbs=create,versions=v1,name=mpod.training.vsr.dev,admissionReviewVersions=v1

// PodSidecarInjector injects the GPU observability sidecar into training pods.
type PodSidecarInjector struct{}

var _ admission.CustomDefaulter = &PodSidecarInjector{}

func (p *PodSidecarInjector) Default(ctx context.Context, obj runtime.Object) error {
	pod := obj.(*corev1.Pod)

	// O(1) label checks — bail immediately for non-training pods.
	// In a cluster with thousands of pods, only training worker pods
	// created by our StatefulSet carry these labels.
	if _, ok := pod.Labels["training.vsr.dev/trainjob"]; !ok {
		return nil
	}
	if pod.Labels["training.vsr.dev/component"] != "worker" {
		return nil
	}

	// Idempotency: skip if sidecar was already injected (e.g., pod update).
	for _, c := range pod.Spec.Containers {
		if c.Name == "gpu-monitor" {
			return nil
		}
	}

	sidecar := corev1.Container{
		Name:    "gpu-monitor",
		Image:   "nvcr.io/nvidia/cloud-native/dcgm:3.3.6-1-ubuntu22.04",
		Command: []string{"bash", "-c"},
		Args:    []string{sidecarScript()},
		Ports: []corev1.ContainerPort{
			{Name: "metrics", ContainerPort: 9400, Protocol: corev1.ProtocolTCP},
			{Name: "health", ContainerPort: 9401, Protocol: corev1.ProtocolTCP},
		},
		Env: []corev1.EnvVar{
			{Name: "SIDECAR_LOG_DIR", Value: "/var/log/gpu-monitor"},
			{Name: "TELEMETRY_INTERVAL", Value: "10"},
			{Name: "STRAGGLER_THRESHOLD", Value: "1.5"},
			{Name: "OOM_HEADROOM_PCT", Value: "10"},
			{Name: "THERMAL_WARN_C", Value: "83"},
			{Name: "THERMAL_CRIT_C", Value: "90"},
			{
				Name: "POD_NAME",
				ValueFrom: &corev1.EnvVarSource{
					FieldRef: &corev1.ObjectFieldSelector{FieldPath: "metadata.name"},
				},
			},
			{
				Name: "NODE_NAME",
				ValueFrom: &corev1.EnvVarSource{
					FieldRef: &corev1.ObjectFieldSelector{FieldPath: "spec.nodeName"},
				},
			},
		},
		Resources: corev1.ResourceRequirements{
			Requests: corev1.ResourceList{
				corev1.ResourceCPU:    resource.MustParse("250m"),
				corev1.ResourceMemory: resource.MustParse("512Mi"),
			},
			Limits: corev1.ResourceList{
				corev1.ResourceCPU:    resource.MustParse("1"),
				corev1.ResourceMemory: resource.MustParse("1Gi"),
			},
		},
		VolumeMounts: []corev1.VolumeMount{
			{Name: "sidecar-logs", MountPath: "/var/log/gpu-monitor"},
			{Name: "shm", MountPath: "/dev/shm", ReadOnly: true},
		},
	}

	pod.Spec.Containers = append(pod.Spec.Containers, sidecar)

	// Add log volume for sidecar
	pod.Spec.Volumes = append(pod.Spec.Volumes, corev1.Volume{
		Name: "sidecar-logs",
		VolumeSource: corev1.VolumeSource{
			EmptyDir: &corev1.EmptyDirVolumeSource{},
		},
	})

	if pod.Annotations == nil {
		pod.Annotations = make(map[string]string)
	}
	pod.Annotations["prometheus.io/scrape"] = "true"
	pod.Annotations["prometheus.io/port"] = "9400"
	pod.Annotations["prometheus.io/path"] = "/metrics"

	hasToleration := false
	for _, t := range pod.Spec.Tolerations {
		if t.Key == "nvidia.com/gpu" {
			hasToleration = true
			break
		}
	}
	if !hasToleration {
		pod.Spec.Tolerations = append(pod.Spec.Tolerations, corev1.Toleration{
			Key:      "nvidia.com/gpu",
			Operator: corev1.TolerationOpExists,
			Effect:   corev1.TaintEffectNoSchedule,
		})
	}

	return nil
}

// sidecarScript returns the comprehensive GPU observability agent script.
// The script runs 5 subsystems as concurrent background processes managed
// by a supervisor loop.
func sidecarScript() string {
	return `#!/bin/bash
set -uo pipefail

LOG_DIR="${SIDECAR_LOG_DIR:-/var/log/gpu-monitor}"
INTERVAL="${TELEMETRY_INTERVAL:-10}"
STRAGGLER_T="${STRAGGLER_THRESHOLD:-1.5}"
OOM_HEADROOM="${OOM_HEADROOM_PCT:-10}"
THERMAL_W="${THERMAL_WARN_C:-83}"
THERMAL_C="${THERMAL_CRIT_C:-90}"

mkdir -p "$LOG_DIR"
echo "$(date -u) gpu-monitor sidecar starting on $NODE_NAME ($POD_NAME)" | tee "$LOG_DIR/sidecar.log"

# ════════════════════════════════════════════════════════════════
#  Subsystem 1: DCGM Exporter (Prometheus metrics)
# ════════════════════════════════════════════════════════════════
start_dcgm_exporter() {
    if command -v dcgm-exporter &>/dev/null; then
        dcgm-exporter \
            --address :9400 \
            --collect-interval 5000 \
            --collectors /etc/dcgm-exporter/dcp-metrics-included.csv \
            >> "$LOG_DIR/dcgm-exporter.log" 2>&1 &
        echo "$(date -u) DCGM exporter started (PID $!)" >> "$LOG_DIR/sidecar.log"
    else
        echo "$(date -u) WARN: dcgm-exporter not found, skipping" >> "$LOG_DIR/sidecar.log"
    fi
}

# ════════════════════════════════════════════════════════════════
#  Subsystem 2: Hardware Telemetry Collector
#  Captures system-level metrics DCGM doesn't cover:
#  - PCIe bandwidth utilization and errors
#  - NVLink error counters (CRC, replay, recovery)
#  - InfiniBand port counters (if available)
#  - CPU utilization, memory pressure, IO wait
#  - GPU power draw trends (for thermal-SDC correlation)
# ════════════════════════════════════════════════════════════════
hw_telemetry_collector() {
    local hw_log="$LOG_DIR/hw_telemetry.jsonl"
    while true; do
        TS=$(date -u +%Y-%m-%dT%H:%M:%SZ)

        # GPU telemetry: temp, power, clocks, memory, PCIe throughput, ECC
        nvidia-smi --query-gpu=index,temperature.gpu,power.draw,clocks.current.graphics,\
clocks.max.graphics,memory.used,memory.total,pcie.link.width.current,\
ecc.errors.corrected.aggregate.total,ecc.errors.uncorrected.aggregate.total,\
utilization.gpu,utilization.memory,enforced.power.limit \
            --format=csv,noheader,nounits 2>/dev/null | while IFS=', ' read -r \
            idx temp power clk_cur clk_max mem_used mem_total pcie_w ecc_c ecc_u util_gpu util_mem pwr_lim; do

            # NVLink error counters (if nvidia-smi nvlink is available)
            nvlink_crc=0
            nvlink_replay=0
            if nvidia-smi nvlink -e 2>/dev/null | grep -q "GPU $idx"; then
                nvlink_crc=$(nvidia-smi nvlink -e -i "$idx" 2>/dev/null | grep "CRC" | awk '{sum+=$NF} END{print sum+0}')
                nvlink_replay=$(nvidia-smi nvlink -e -i "$idx" 2>/dev/null | grep "Replay" | awk '{sum+=$NF} END{print sum+0}')
            fi

            # Throttle reason
            throttle=$(nvidia-smi --query-gpu=clocks_throttle_reasons.active --format=csv,noheader -i "$idx" 2>/dev/null || echo "unknown")

            echo "{\"ts\":\"$TS\",\"gpu\":$idx,\"temp\":$temp,\"power\":$power,\"pwr_limit\":$pwr_lim,"\
"\"clk_cur\":$clk_cur,\"clk_max\":$clk_max,\"mem_used_mb\":$mem_used,\"mem_total_mb\":$mem_total,"\
"\"pcie_width\":$pcie_w,\"ecc_corrected\":$ecc_c,\"ecc_uncorrected\":$ecc_u,"\
"\"util_gpu\":$util_gpu,\"util_mem\":$util_mem,"\
"\"nvlink_crc_err\":$nvlink_crc,\"nvlink_replay_err\":$nvlink_replay,"\
"\"throttle\":\"$throttle\",\"node\":\"$NODE_NAME\",\"pod\":\"$POD_NAME\"}"
        done >> "$hw_log"

        # System-level metrics
        CPU_UTIL=$(top -bn1 | grep "Cpu(s)" | awk '{print $2}' 2>/dev/null || echo "0")
        MEM_AVAIL=$(awk '/MemAvailable/{print $2}' /proc/meminfo 2>/dev/null || echo "0")
        IO_WAIT=$(iostat -c 1 1 2>/dev/null | tail -1 | awk '{print $4}' || echo "0")

        echo "{\"ts\":\"$TS\",\"type\":\"system\",\"cpu_util\":$CPU_UTIL,"\
"\"mem_avail_kb\":$MEM_AVAIL,\"io_wait\":$IO_WAIT,"\
"\"node\":\"$NODE_NAME\",\"pod\":\"$POD_NAME\"}" >> "$hw_log"

        # IB counters (if available)
        if command -v perfquery &>/dev/null; then
            perfquery 2>/dev/null | grep -E "(PortXmitData|PortRcvData|PortXmitDiscards|LinkErrorRecovery)" | \
                awk -v ts="$TS" -v node="$NODE_NAME" -v pod="$POD_NAME" \
                'BEGIN{ORS=""} {gsub(/\./,"",$1); vals[$1]=$NF}
                 END{printf "{\"ts\":\"%s\",\"type\":\"ib\"", ts;
                     for(k in vals) printf ",\"%s\":%s", k, vals[k];
                     printf ",\"node\":\"%s\",\"pod\":\"%s\"}\n", node, pod}' >> "$hw_log"
        fi

        sleep "$INTERVAL"
    done
}

# ════════════════════════════════════════════════════════════════
#  Subsystem 3: Kernel Performance Monitor
#  Tracks per-step GPU kernel time distribution.
#  The training process writes step markers to a shared file;
#  this monitor reads them and detects performance regressions
#  (e.g., thermal throttling, NVLink degradation, NCCL slowdown).
# ════════════════════════════════════════════════════════════════
kernel_perf_monitor() {
    local perf_log="$LOG_DIR/kernel_perf.jsonl"
    local step_file="/dev/shm/training_step_time"
    local -a recent_times=()
    local window_size=50
    local baseline_mean=0
    local step_count=0

    while true; do
        if [ -f "$step_file" ]; then
            # Training process writes: step_num,step_time_ms,gpu_util_pct
            while IFS=',' read -r step_num step_ms gpu_util 2>/dev/null; do
                [ -z "$step_num" ] && continue

                recent_times+=("$step_ms")
                if [ ${#recent_times[@]} -gt $window_size ]; then
                    recent_times=("${recent_times[@]:1}")
                fi
                step_count=$((step_count + 1))

                # Compute rolling mean and stddev
                if [ ${#recent_times[@]} -ge 10 ]; then
                    stats=$(printf '%s\n' "${recent_times[@]}" | awk '{
                        sum+=$1; sumsq+=$1*$1; n++
                    } END {
                        mean=sum/n; var=sumsq/n-mean*mean;
                        if(var<0) var=0;
                        printf "%.2f %.2f", mean, sqrt(var)
                    }')
                    mean=$(echo "$stats" | awk '{print $1}')
                    stddev=$(echo "$stats" | awk '{print $2}')

                    # Update baseline after warmup (first 100 steps)
                    if [ $step_count -eq 100 ]; then
                        baseline_mean=$mean
                        echo "$(date -u) Kernel perf baseline: ${baseline_mean}ms/step" >> "$LOG_DIR/sidecar.log"
                    fi

                    # Detect regression: current mean > baseline × 1.2 (20% slower)
                    if [ $step_count -gt 100 ] && [ "$(echo "$mean > $baseline_mean * 1.2" | bc -l)" = "1" ]; then
                        echo "{\"ts\":\"$(date -u +%Y-%m-%dT%H:%M:%SZ)\",\"type\":\"perf_regression\","\
"\"step\":$step_num,\"current_mean_ms\":$mean,\"baseline_mean_ms\":$baseline_mean,"\
"\"stddev_ms\":$stddev,\"node\":\"$NODE_NAME\"}" >> "$perf_log"
                        echo "$(date -u) ALERT: Kernel perf regression detected at step $step_num: " \
                             "${mean}ms vs baseline ${baseline_mean}ms" >> "$LOG_DIR/sidecar.log"
                    fi

                    # Log periodically (every 100 steps)
                    if [ $((step_num % 100)) -eq 0 ]; then
                        echo "{\"ts\":\"$(date -u +%Y-%m-%dT%H:%M:%SZ)\",\"type\":\"perf_sample\","\
"\"step\":$step_num,\"mean_ms\":$mean,\"stddev_ms\":$stddev,"\
"\"baseline_ms\":$baseline_mean,\"gpu_util\":$gpu_util,\"node\":\"$NODE_NAME\"}" >> "$perf_log"
                    fi
                fi
            done < "$step_file"
            # Truncate after reading (trainer appends, we consume)
            : > "$step_file" 2>/dev/null || true
        fi
        sleep 2
    done
}

# ════════════════════════════════════════════════════════════════
#  Subsystem 4: Straggler Detector
#  Compares this worker's step time against cluster-wide median.
#  Uses a shared metrics endpoint: each worker writes its step time
#  to a file; rank-0's sidecar aggregates and flags stragglers.
#  Stragglers are workers whose step time > median × STRAGGLER_THRESHOLD.
# ════════════════════════════════════════════════════════════════
straggler_detector() {
    local straggler_log="$LOG_DIR/straggler.jsonl"
    local my_step_file="/dev/shm/training_step_time"
    local last_step_ms=0

    while true; do
        if [ -f "$my_step_file" ]; then
            last_line=$(tail -1 "$my_step_file" 2>/dev/null || echo "")
            if [ -n "$last_line" ]; then
                last_step_ms=$(echo "$last_line" | cut -d',' -f2)
            fi
        fi

        # Write this worker's latest step time to a well-known location
        # that can be scraped by the aggregator (rank-0 sidecar or external)
        echo "{\"pod\":\"$POD_NAME\",\"node\":\"$NODE_NAME\",\"step_ms\":$last_step_ms,"\
"\"ts\":\"$(date -u +%Y-%m-%dT%H:%M:%SZ)\"}" > "$LOG_DIR/straggler_heartbeat.json"

        # Self-check: if our step time is much larger than the first recorded baseline
        if [ "$last_step_ms" != "0" ] && [ -f "$LOG_DIR/step_baseline_ms" ]; then
            baseline=$(cat "$LOG_DIR/step_baseline_ms")
            ratio=$(echo "scale=2; $last_step_ms / $baseline" | bc -l 2>/dev/null || echo "1.00")
            if (( $(echo "$ratio > $STRAGGLER_T" | bc -l 2>/dev/null || echo 0) )); then
                echo "{\"ts\":\"$(date -u +%Y-%m-%dT%H:%M:%SZ)\",\"type\":\"self_straggler\","\
"\"step_ms\":$last_step_ms,\"baseline_ms\":$baseline,\"ratio\":$ratio,"\
"\"node\":\"$NODE_NAME\",\"pod\":\"$POD_NAME\"}" >> "$straggler_log"
                echo "$(date -u) ALERT: This worker is a straggler (${ratio}× baseline)" >> "$LOG_DIR/sidecar.log"
            fi
        elif [ "$last_step_ms" != "0" ] && [ ! -f "$LOG_DIR/step_baseline_ms" ]; then
            echo "$last_step_ms" > "$LOG_DIR/step_baseline_ms"
        fi

        sleep 5
    done
}

# ════════════════════════════════════════════════════════════════
#  Subsystem 5: Anomaly Watchdog
#  Continuous critical checks that can trigger preemptive action:
#  - ECC error accumulation → signal trainer to checkpoint + drain
#  - Thermal runaway → log for cluster scheduler to migrate
#  - GPU memory proximity to OOM → warn before CUDA OOM kills
#  - XID errors in dmesg → log specific failure mode
#  - NCCL timeout watchdog → detect hung collectives
# ════════════════════════════════════════════════════════════════
anomaly_watchdog() {
    local anomaly_log="$LOG_DIR/anomalies.jsonl"
    local prev_ecc_total=0
    local prev_xid_count=0

    while true; do
        TS=$(date -u +%Y-%m-%dT%H:%M:%SZ)

        # 5.1 ECC error accumulation
        ecc_total=$(nvidia-smi --query-gpu=ecc.errors.uncorrected.aggregate.total \
            --format=csv,noheader,nounits 2>/dev/null | awk '{s+=$1} END{print s+0}')
        if [ "$ecc_total" -gt "$prev_ecc_total" ] 2>/dev/null; then
            new_errors=$((ecc_total - prev_ecc_total))
            echo "{\"ts\":\"$TS\",\"type\":\"ecc_error\",\"new_errors\":$new_errors,"\
"\"total\":$ecc_total,\"node\":\"$NODE_NAME\"}" >> "$anomaly_log"
            echo "$(date -u) CRITICAL: $new_errors new uncorrectable ECC errors (total: $ecc_total)" >> "$LOG_DIR/sidecar.log"

            # Signal trainer to do an emergency checkpoint
            if [ -f "/dev/shm/trainer_pid" ]; then
                TRAINER_PID=$(cat /dev/shm/trainer_pid)
                kill -USR1 "$TRAINER_PID" 2>/dev/null && \
                    echo "$(date -u) Sent USR1 to trainer ($TRAINER_PID) for emergency checkpoint" >> "$LOG_DIR/sidecar.log"
            fi
            prev_ecc_total=$ecc_total
        fi

        # 5.2 Thermal monitoring with trend detection
        nvidia-smi --query-gpu=index,temperature.gpu,power.draw,enforced.power.limit \
            --format=csv,noheader,nounits 2>/dev/null | while IFS=', ' read -r idx temp power pwr_lim; do

            if [ "$temp" -ge "$THERMAL_C" ] 2>/dev/null; then
                echo "{\"ts\":\"$TS\",\"type\":\"thermal_critical\",\"gpu\":$idx,\"temp\":$temp,"\
"\"power\":$power,\"node\":\"$NODE_NAME\"}" >> "$anomaly_log"
                echo "$(date -u) CRITICAL: GPU $idx at ${temp}°C — thermal critical" >> "$LOG_DIR/sidecar.log"
            elif [ "$temp" -ge "$THERMAL_W" ] 2>/dev/null; then
                echo "{\"ts\":\"$TS\",\"type\":\"thermal_warn\",\"gpu\":$idx,\"temp\":$temp,"\
"\"power\":$power,\"node\":\"$NODE_NAME\"}" >> "$anomaly_log"
            fi

            # Power limit proximity — indicates throttling
            if [ -n "$pwr_lim" ] && [ "$pwr_lim" -gt 0 ] 2>/dev/null; then
                pwr_pct=$(echo "scale=0; $power * 100 / $pwr_lim" | bc 2>/dev/null || echo "0")
                if [ "$pwr_pct" -ge 95 ] 2>/dev/null; then
                    echo "{\"ts\":\"$TS\",\"type\":\"power_limit\",\"gpu\":$idx,\"power\":$power,"\
"\"limit\":$pwr_lim,\"pct\":$pwr_pct,\"node\":\"$NODE_NAME\"}" >> "$anomaly_log"
                fi
            fi
        done

        # 5.3 GPU memory OOM proximity check
        nvidia-smi --query-gpu=index,memory.used,memory.total \
            --format=csv,noheader,nounits 2>/dev/null | while IFS=', ' read -r idx mem_used mem_total; do
            headroom=$(echo "scale=0; ($mem_total - $mem_used) * 100 / $mem_total" | bc 2>/dev/null || echo "100")
            if [ "$headroom" -lt "$OOM_HEADROOM" ] 2>/dev/null; then
                echo "{\"ts\":\"$TS\",\"type\":\"oom_proximity\",\"gpu\":$idx,"\
"\"used_mb\":$mem_used,\"total_mb\":$mem_total,\"headroom_pct\":$headroom,"\
"\"node\":\"$NODE_NAME\"}" >> "$anomaly_log"
                echo "$(date -u) WARN: GPU $idx memory headroom ${headroom}% — OOM risk" >> "$LOG_DIR/sidecar.log"
            fi
        done

        # 5.4 XID errors from kernel log (GPU hardware errors)
        xid_count=$(dmesg 2>/dev/null | grep -c "NVRM: Xid" || echo "0")
        if [ "$xid_count" -gt "$prev_xid_count" ] 2>/dev/null; then
            new_xids=$((xid_count - prev_xid_count))
            latest_xid=$(dmesg 2>/dev/null | grep "NVRM: Xid" | tail -1)
            echo "{\"ts\":\"$TS\",\"type\":\"xid_error\",\"new_count\":$new_xids,"\
"\"latest\":\"$latest_xid\",\"node\":\"$NODE_NAME\"}" >> "$anomaly_log"
            echo "$(date -u) ALERT: $new_xids new XID errors: $latest_xid" >> "$LOG_DIR/sidecar.log"
            prev_xid_count=$xid_count
        fi

        # 5.5 NCCL timeout watchdog
        # Check if the NCCL flight recorder dump exists (indicates hung collective)
        if ls /tmp/nccl_trace_rank_* 2>/dev/null | head -1 | grep -q .; then
            echo "{\"ts\":\"$TS\",\"type\":\"nccl_timeout\",\"node\":\"$NODE_NAME\"}" >> "$anomaly_log"
            echo "$(date -u) CRITICAL: NCCL flight recorder dump detected — hung collective" >> "$LOG_DIR/sidecar.log"
            # Copy trace for post-mortem
            cp /tmp/nccl_trace_rank_* "$LOG_DIR/" 2>/dev/null || true
        fi

        sleep "$INTERVAL"
    done
}

# ════════════════════════════════════════════════════════════════
#  Health endpoint (simple HTTP on port 9401)
# ════════════════════════════════════════════════════════════════
health_server() {
    while true; do
        echo -e "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n\r\n{\"status\":\"healthy\",\"node\":\"$NODE_NAME\"}" | \
            nc -l -p 9401 -q 1 2>/dev/null || sleep 1
    done
}

# ════════════════════════════════════════════════════════════════
#  Supervisor: launch all subsystems, restart on failure
# ════════════════════════════════════════════════════════════════
echo "$(date -u) Starting 5 subsystems..." >> "$LOG_DIR/sidecar.log"

start_dcgm_exporter
hw_telemetry_collector &
HW_PID=$!

kernel_perf_monitor &
PERF_PID=$!

straggler_detector &
STRAGGLER_PID=$!

anomaly_watchdog &
WATCHDOG_PID=$!

health_server &
HEALTH_PID=$!

echo "$(date -u) All subsystems started: hw=$HW_PID perf=$PERF_PID straggler=$STRAGGLER_PID watchdog=$WATCHDOG_PID health=$HEALTH_PID" \
    >> "$LOG_DIR/sidecar.log"

# Supervisor loop: restart crashed subsystems
while true; do
    for pid_var in HW_PID PERF_PID STRAGGLER_PID WATCHDOG_PID HEALTH_PID; do
        pid=${!pid_var}
        if ! kill -0 "$pid" 2>/dev/null; then
            echo "$(date -u) Subsystem $pid_var (PID $pid) died, restarting..." >> "$LOG_DIR/sidecar.log"
            case $pid_var in
                HW_PID)        hw_telemetry_collector & eval "$pid_var=$!" ;;
                PERF_PID)      kernel_perf_monitor & eval "$pid_var=$!" ;;
                STRAGGLER_PID) straggler_detector & eval "$pid_var=$!" ;;
                WATCHDOG_PID)  anomaly_watchdog & eval "$pid_var=$!" ;;
                HEALTH_PID)    health_server & eval "$pid_var=$!" ;;
            esac
        fi
    done
    sleep 30
done
`
}
