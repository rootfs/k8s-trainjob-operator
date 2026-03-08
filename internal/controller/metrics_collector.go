package controller

import (
	"fmt"

	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"

	aiv1 "github.com/rootfs/trainjob-operator/api/v1alpha1"
)

const (
	defaultMetricsFilePath   = "/var/run/training/metrics.json"
	defaultScrapeIntervalSec = "30"
	defaultPrometheusPort    = 9402
	metricsVolumeName        = "training-metrics"
	metricsVolumePath        = "/var/run/training"
)

func metricsCollectorEnabled(tj *aiv1.TrainJob) bool {
	return tj.Spec.MetricsConfig != nil && tj.Spec.MetricsConfig.Enabled
}

func metricsFilePath(tj *aiv1.TrainJob) string {
	if tj.Spec.MetricsConfig != nil && tj.Spec.MetricsConfig.MetricsFilePath != "" {
		return tj.Spec.MetricsConfig.MetricsFilePath
	}
	return defaultMetricsFilePath
}

// buildMetricsCollectorSidecar returns the metrics-collector sidecar container.
// It reads a JSON metrics file written by the training script on a shared emptyDir
// and exposes the metrics in Prometheus format on a configurable port. It also
// periodically patches the TrainJob status sub-resource with structured training
// metrics so agents can consume them without scraping Prometheus.
func buildMetricsCollectorSidecar(tj *aiv1.TrainJob) corev1.Container {
	scrapeInterval := defaultScrapeIntervalSec
	if tj.Spec.MetricsConfig != nil && tj.Spec.MetricsConfig.ScrapeIntervalSeconds != nil {
		scrapeInterval = intToStr(int(*tj.Spec.MetricsConfig.ScrapeIntervalSeconds))
	}

	promPort := int32(defaultPrometheusPort)
	if tj.Spec.MetricsConfig != nil && tj.Spec.MetricsConfig.PrometheusPort != nil {
		promPort = *tj.Spec.MetricsConfig.PrometheusPort
	}

	return corev1.Container{
		Name:    "metrics-collector",
		Image:   "ghcr.io/rootfs/trainjob-metrics-collector:latest",
		Command: []string{"bash", "-c"},
		Args:    []string{metricsCollectorScript()},
		Env: []corev1.EnvVar{
			{Name: "METRICS_FILE", Value: metricsFilePath(tj)},
			{Name: "SCRAPE_INTERVAL", Value: scrapeInterval},
			{Name: "PROM_PORT", Value: intToStr(int(promPort))},
			{Name: "TRAINJOB_NAME", Value: tj.Name},
			{Name: "TRAINJOB_NAMESPACE", Value: tj.Namespace},
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
		Ports: []corev1.ContainerPort{
			{Name: "train-metrics", ContainerPort: promPort, Protocol: corev1.ProtocolTCP},
		},
		Resources: corev1.ResourceRequirements{
			Requests: corev1.ResourceList{
				corev1.ResourceCPU:    resource.MustParse("100m"),
				corev1.ResourceMemory: resource.MustParse("128Mi"),
			},
			Limits: corev1.ResourceList{
				corev1.ResourceCPU:    resource.MustParse("500m"),
				corev1.ResourceMemory: resource.MustParse("256Mi"),
			},
		},
		VolumeMounts: []corev1.VolumeMount{
			{Name: metricsVolumeName, MountPath: metricsVolumePath},
		},
	}
}

// metricsCollectorScript is the sidecar startup script. It reads a JSON file
// written by the training process and exposes the data as Prometheus metrics.
//
// The training script writes to $METRICS_FILE with this schema:
//
//	{
//	  "step": 12400,
//	  "total_steps": 50000,
//	  "train_loss": 0.342,
//	  "val_loss": 0.389,
//	  "gradient_norm": 1.24,
//	  "gradient_norm_max": 3.8,
//	  "zero_gradient_pct": 0.02,
//	  "tokens_per_second": 48200,
//	  "samples_per_second": 12.4,
//	  "mfu": 0.41,
//	  "comm_compute_ratio": 0.28,
//	  "step_time_sec": 1.82,
//	  "peak_memory_gb": 74.2,
//	  "allocated_memory_gb": 68.1,
//	  "oom_events": 0,
//	  "ecc_errors": 0,
//	  "thermal_events": 0,
//	  "nvlink_errors": 0
//	}
func metricsCollectorScript() string {
	return `#!/bin/bash
set -uo pipefail

METRICS_FILE="${METRICS_FILE:-/var/run/training/metrics.json}"
SCRAPE_INTERVAL="${SCRAPE_INTERVAL:-30}"
PROM_PORT="${PROM_PORT:-9402}"

echo "$(date -u) metrics-collector starting: file=$METRICS_FILE interval=${SCRAPE_INTERVAL}s port=$PROM_PORT"

# Wait for metrics file to appear
while [ ! -f "$METRICS_FILE" ]; do
    sleep 5
done

echo "$(date -u) metrics file detected, starting collection"

# Prometheus metrics exporter loop
serve_metrics() {
    while true; do
        if [ -f "$METRICS_FILE" ]; then
            STEP=$(jq -r '.step // 0' "$METRICS_FILE" 2>/dev/null)
            TOTAL=$(jq -r '.total_steps // 0' "$METRICS_FILE" 2>/dev/null)
            TRAIN_LOSS=$(jq -r '.train_loss // 0' "$METRICS_FILE" 2>/dev/null)
            VAL_LOSS=$(jq -r '.val_loss // 0' "$METRICS_FILE" 2>/dev/null)
            GRAD_NORM=$(jq -r '.gradient_norm // 0' "$METRICS_FILE" 2>/dev/null)
            GRAD_MAX=$(jq -r '.gradient_norm_max // 0' "$METRICS_FILE" 2>/dev/null)
            ZERO_GRAD=$(jq -r '.zero_gradient_pct // 0' "$METRICS_FILE" 2>/dev/null)
            TPS=$(jq -r '.tokens_per_second // 0' "$METRICS_FILE" 2>/dev/null)
            SPS=$(jq -r '.samples_per_second // 0' "$METRICS_FILE" 2>/dev/null)
            MFU=$(jq -r '.mfu // 0' "$METRICS_FILE" 2>/dev/null)
            COMM=$(jq -r '.comm_compute_ratio // 0' "$METRICS_FILE" 2>/dev/null)
            STEP_TIME=$(jq -r '.step_time_sec // 0' "$METRICS_FILE" 2>/dev/null)
            PEAK_MEM=$(jq -r '.peak_memory_gb // 0' "$METRICS_FILE" 2>/dev/null)
            ALLOC_MEM=$(jq -r '.allocated_memory_gb // 0' "$METRICS_FILE" 2>/dev/null)
            OOM=$(jq -r '.oom_events // 0' "$METRICS_FILE" 2>/dev/null)
            ECC=$(jq -r '.ecc_errors // 0' "$METRICS_FILE" 2>/dev/null)
            THERMAL=$(jq -r '.thermal_events // 0' "$METRICS_FILE" 2>/dev/null)
            NVLINK=$(jq -r '.nvlink_errors // 0' "$METRICS_FILE" 2>/dev/null)

            BODY="# HELP trainjob_current_step Current training step
# TYPE trainjob_current_step gauge
trainjob_current_step{trainjob=\"$TRAINJOB_NAME\",pod=\"$POD_NAME\",node=\"$NODE_NAME\"} $STEP
# HELP trainjob_total_steps Total expected training steps
# TYPE trainjob_total_steps gauge
trainjob_total_steps{trainjob=\"$TRAINJOB_NAME\"} $TOTAL
# HELP trainjob_train_loss Training loss
# TYPE trainjob_train_loss gauge
trainjob_train_loss{trainjob=\"$TRAINJOB_NAME\",pod=\"$POD_NAME\"} $TRAIN_LOSS
# HELP trainjob_val_loss Validation loss
# TYPE trainjob_val_loss gauge
trainjob_val_loss{trainjob=\"$TRAINJOB_NAME\",pod=\"$POD_NAME\"} $VAL_LOSS
# HELP trainjob_gradient_norm Gradient L2 norm
# TYPE trainjob_gradient_norm gauge
trainjob_gradient_norm{trainjob=\"$TRAINJOB_NAME\",pod=\"$POD_NAME\"} $GRAD_NORM
# HELP trainjob_gradient_norm_max Max gradient norm across parameters
# TYPE trainjob_gradient_norm_max gauge
trainjob_gradient_norm_max{trainjob=\"$TRAINJOB_NAME\",pod=\"$POD_NAME\"} $GRAD_MAX
# HELP trainjob_zero_gradient_pct Fraction of parameters with zero gradients
# TYPE trainjob_zero_gradient_pct gauge
trainjob_zero_gradient_pct{trainjob=\"$TRAINJOB_NAME\",pod=\"$POD_NAME\"} $ZERO_GRAD
# HELP trainjob_tokens_per_second Training throughput in tokens/sec
# TYPE trainjob_tokens_per_second gauge
trainjob_tokens_per_second{trainjob=\"$TRAINJOB_NAME\",pod=\"$POD_NAME\"} $TPS
# HELP trainjob_samples_per_second Training throughput in samples/sec
# TYPE trainjob_samples_per_second gauge
trainjob_samples_per_second{trainjob=\"$TRAINJOB_NAME\",pod=\"$POD_NAME\"} $SPS
# HELP trainjob_mfu Model FLOPS utilization
# TYPE trainjob_mfu gauge
trainjob_mfu{trainjob=\"$TRAINJOB_NAME\",pod=\"$POD_NAME\"} $MFU
# HELP trainjob_comm_compute_ratio Fraction of step time in NCCL collectives
# TYPE trainjob_comm_compute_ratio gauge
trainjob_comm_compute_ratio{trainjob=\"$TRAINJOB_NAME\",pod=\"$POD_NAME\"} $COMM
# HELP trainjob_step_time_seconds Time per training step
# TYPE trainjob_step_time_seconds gauge
trainjob_step_time_seconds{trainjob=\"$TRAINJOB_NAME\",pod=\"$POD_NAME\"} $STEP_TIME
# HELP trainjob_peak_memory_gb Peak GPU memory usage in GB
# TYPE trainjob_peak_memory_gb gauge
trainjob_peak_memory_gb{trainjob=\"$TRAINJOB_NAME\",pod=\"$POD_NAME\"} $PEAK_MEM
# HELP trainjob_allocated_memory_gb Allocated GPU memory in GB
# TYPE trainjob_allocated_memory_gb gauge
trainjob_allocated_memory_gb{trainjob=\"$TRAINJOB_NAME\",pod=\"$POD_NAME\"} $ALLOC_MEM
# HELP trainjob_oom_events_total Number of OOM events
# TYPE trainjob_oom_events_total counter
trainjob_oom_events_total{trainjob=\"$TRAINJOB_NAME\",pod=\"$POD_NAME\"} $OOM
# HELP trainjob_ecc_errors_total Number of ECC memory errors
# TYPE trainjob_ecc_errors_total counter
trainjob_ecc_errors_total{trainjob=\"$TRAINJOB_NAME\",pod=\"$POD_NAME\"} $ECC
# HELP trainjob_thermal_events_total Number of thermal throttle events
# TYPE trainjob_thermal_events_total counter
trainjob_thermal_events_total{trainjob=\"$TRAINJOB_NAME\",pod=\"$POD_NAME\"} $THERMAL
# HELP trainjob_nvlink_errors_total Number of NVLink errors
# TYPE trainjob_nvlink_errors_total counter
trainjob_nvlink_errors_total{trainjob=\"$TRAINJOB_NAME\",pod=\"$POD_NAME\"} $NVLINK"

            RESPONSE="HTTP/1.1 200 OK\r\nContent-Type: text/plain; version=0.0.4\r\nContent-Length: ${#BODY}\r\n\r\n${BODY}"
            echo -ne "$RESPONSE" | nc -l -p "$PROM_PORT" -q 1 2>/dev/null || sleep 1
        else
            sleep "$SCRAPE_INTERVAL"
        fi
    done
}

serve_metrics &

# Status patcher loop: read metrics.json and patch TrainJob status via Kubernetes API.
# Uses the service account token mounted into the pod. Only rank 0 patches status
# to avoid conflicting writes.
if echo "$POD_NAME" | grep -q '\-0$'; then
    echo "$(date -u) rank-0 pod detected, enabling status patching"
    TOKEN_PATH="/var/run/secrets/kubernetes.io/serviceaccount/token"
    CA_PATH="/var/run/secrets/kubernetes.io/serviceaccount/ca.crt"
    API_SERVER="https://kubernetes.default.svc"

    while true; do
        sleep "$SCRAPE_INTERVAL"
        if [ ! -f "$METRICS_FILE" ]; then
            continue
        fi

        # Read metrics and build JSON patch for status.training
        PATCH=$(jq -c '{
            "status": {
                "training": {
                    "trainLoss": .train_loss,
                    "valLoss": .val_loss,
                    "gradientNorm": .gradient_norm,
                    "gradientNormMax": .gradient_norm_max,
                    "zeroGradientPct": .zero_gradient_pct,
                    "tokensPerSecond": .tokens_per_second,
                    "samplesPerSecond": .samples_per_second,
                    "mfu": .mfu,
                    "commComputeRatio": .comm_compute_ratio,
                    "stepTimeSec": .step_time_sec,
                    "peakMemoryGB": .peak_memory_gb,
                    "allocatedGB": .allocated_memory_gb,
                    "oomEvents": (.oom_events // 0),
                    "eccErrors": (.ecc_errors // 0),
                    "thermalEvents": (.thermal_events // 0),
                    "nvlinkErrors": (.nvlink_errors // 0)
                },
                "currentStep": (.step // 0),
                "totalSteps": (.total_steps // 0)
            }
        }' "$METRICS_FILE" 2>/dev/null)

        if [ -n "$PATCH" ] && [ "$PATCH" != "null" ]; then
            curl -s --cacert "$CA_PATH" \
                -H "Authorization: Bearer $(cat $TOKEN_PATH)" \
                -H "Content-Type: application/merge-patch+json" \
                -X PATCH \
                "${API_SERVER}/apis/training.vsr.dev/v1alpha1/namespaces/${TRAINJOB_NAMESPACE}/trainjobs/${TRAINJOB_NAME}/status" \
                -d "$PATCH" > /dev/null 2>&1
        fi
    done
else
    echo "$(date -u) non-rank-0 pod, status patching disabled (Prometheus export only)"
    wait
fi
`
}

func intToStr(v int) string {
	return fmt.Sprintf("%d", v)
}
