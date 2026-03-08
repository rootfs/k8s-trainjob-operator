package controller

import (
	"fmt"
	"strconv"

	appsv1 "k8s.io/api/apps/v1"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

	aiv1 "github.com/rootfs/trainjob-operator/api/v1alpha1"
)

// buildHeadlessService creates the headless Service that provides stable DNS
// names for each worker pod: <trainjob>-workers-0.<trainjob>-headless.<ns>.svc
// This is required by torch.distributed for rendezvous.
func buildHeadlessService(tj *aiv1.TrainJob) *corev1.Service {
	return &corev1.Service{
		ObjectMeta: metav1.ObjectMeta{
			Name:      headlessSvcName(tj),
			Namespace: tj.Namespace,
			Labels: map[string]string{
				"training.vsr.dev/trainjob":  tj.Name,
				"training.vsr.dev/component": "workers",
			},
		},
		Spec: corev1.ServiceSpec{
			// Headless service: ClusterIP = None
			ClusterIP: corev1.ClusterIPNone,
			Selector: map[string]string{
				"training.vsr.dev/trainjob":  tj.Name,
				"training.vsr.dev/component": "worker",
			},
			Ports: []corev1.ServicePort{
				{
					Name:     "nccl",
					Port:     29500,
					Protocol: corev1.ProtocolTCP,
				},
			},
			// Enable pod DNS records immediately (don't wait for readiness)
			PublishNotReadyAddresses: true,
		},
	}
}

// buildWorkerStatefulSet creates the StatefulSet for training workers.
// StatefulSet (not Deployment) because:
//   - Stable pod names: <name>-0, <name>-1, ... (needed for RANK assignment)
//   - Ordered startup: rank 0 starts first (rendezvous coordinator)
//   - Stable DNS: each pod gets a DNS entry via the headless service
func buildWorkerStatefulSet(tj *aiv1.TrainJob) *appsv1.StatefulSet {
	totalGPUs := tj.Spec.NumNodes * tj.Spec.GPUsPerNode
	cpDegree := tj.Spec.CPDegree
	if cpDegree == 0 {
		cpDegree = 1
	}
	fsdpDegree := totalGPUs / (tj.Spec.TPDegree * tj.Spec.PPDegree * cpDegree)

	labels := map[string]string{
		"training.vsr.dev/trainjob":  tj.Name,
		"training.vsr.dev/component": "worker",
		"training.vsr.dev/model":     tj.Spec.Model,
	}

	envVars := []corev1.EnvVar{
		// PyTorch distributed environment
		{Name: "MASTER_ADDR", Value: fmt.Sprintf("%s-workers-0.%s.%s.svc", tj.Name, headlessSvcName(tj), tj.Namespace)},
		{Name: "MASTER_PORT", Value: "29500"},
		{Name: "WORLD_SIZE", Value: strconv.Itoa(int(totalGPUs))},
		{Name: "NNODES", Value: strconv.Itoa(int(tj.Spec.NumNodes))},
		{Name: "NPROC_PER_NODE", Value: strconv.Itoa(int(tj.Spec.GPUsPerNode))},
		// Rank is derived from pod ordinal index (set by StatefulSet)
		// The entrypoint script computes: RANK = POD_INDEX * NPROC_PER_NODE
		{
			Name: "POD_INDEX",
			ValueFrom: &corev1.EnvVarSource{
				FieldRef: &corev1.ObjectFieldSelector{
					FieldPath: "metadata.labels['apps.kubernetes.io/pod-index']",
				},
			},
		},
		{
			Name: "NODE_NAME",
			ValueFrom: &corev1.EnvVarSource{
				FieldRef: &corev1.ObjectFieldSelector{FieldPath: "spec.nodeName"},
			},
		},

		// Parallelism config
		{Name: "TP_DEGREE", Value: strconv.Itoa(int(tj.Spec.TPDegree))},
		{Name: "PP_DEGREE", Value: strconv.Itoa(int(tj.Spec.PPDegree))},
		{Name: "FSDP_DEGREE", Value: strconv.Itoa(int(fsdpDegree))},
		{Name: "CP_DEGREE", Value: strconv.Itoa(int(cpDegree))},
		{Name: "PRECISION", Value: tj.Spec.Precision},

		// Checkpoint config
		{Name: "CHECKPOINT_DIR", Value: tj.Spec.Checkpoint.StoragePath},

		// NCCL flight recorder for post-mortem debugging
		{Name: "TORCH_NCCL_TRACE_BUFFER_SIZE", Value: "100"},
		{Name: "TORCH_NCCL_DUMP_ON_TIMEOUT", Value: "1"},

		// OOM debugging
		{Name: "PYTORCH_CUDA_ALLOC_CONF", Value: "expandable_segments:True"},
	}

	// Append checkpoint interval if set
	if tj.Spec.Checkpoint.IntervalMinutes != nil {
		envVars = append(envVars, corev1.EnvVar{
			Name:  "CHECKPOINT_INTERVAL_MINUTES",
			Value: strconv.Itoa(int(*tj.Spec.Checkpoint.IntervalMinutes)),
		})
	}

	// Append user-defined env vars
	for _, e := range tj.Spec.Env {
		envVars = append(envVars, corev1.EnvVar{Name: e.Name, Value: e.Value})
	}

	// Resource requests
	cpuReq := tj.Spec.CPUPerNode
	if cpuReq.IsZero() {
		cpuReq = resource.MustParse("32")
	}
	memReq := tj.Spec.MemPerNode
	if memReq.IsZero() {
		memReq = resource.MustParse("256Gi")
	}

	// Container definition
	container := corev1.Container{
		Name:    "trainer",
		Image:   tj.Spec.Image,
		Command: tj.Spec.Command,
		Args:    tj.Spec.Args,
		Env:     envVars,
		Ports: []corev1.ContainerPort{
			{Name: "nccl", ContainerPort: 29500, Protocol: corev1.ProtocolTCP},
		},
		Resources: corev1.ResourceRequirements{
			Requests: corev1.ResourceList{
				corev1.ResourceCPU:    cpuReq,
				corev1.ResourceMemory: memReq,
				"nvidia.com/gpu":      resource.MustParse(fmt.Sprintf("%d", tj.Spec.GPUsPerNode)),
			},
			Limits: corev1.ResourceList{
				"nvidia.com/gpu": resource.MustParse(fmt.Sprintf("%d", tj.Spec.GPUsPerNode)),
			},
		},
		VolumeMounts: []corev1.VolumeMount{
			{Name: "checkpoints", MountPath: tj.Spec.Checkpoint.StoragePath},
			{Name: "shm", MountPath: "/dev/shm"},
		},
	}

	// Build container list: trainer + optional sidecars.
	// Sidecars are included directly in the StatefulSet pod template instead
	// of being injected via a cluster-wide pod webhook. This eliminates the
	// need for a MutatingWebhookConfiguration on pods and makes the operator
	// fully self-service (no cluster-scoped resources needed).
	containers := []corev1.Container{container}

	sidecarEnabled := tj.Spec.EnableSidecar == nil || *tj.Spec.EnableSidecar
	if sidecarEnabled {
		sidecar := buildGPUMonitorSidecar()
		containers = append(containers, sidecar)
	}

	mcEnabled := metricsCollectorEnabled(tj)
	if mcEnabled {
		mc := buildMetricsCollectorSidecar(tj)
		containers = append(containers, mc)

		// The trainer container also needs the shared volume mount for writing metrics
		containers[0].VolumeMounts = append(containers[0].VolumeMounts,
			corev1.VolumeMount{Name: metricsVolumeName, MountPath: metricsVolumePath})
		containers[0].Env = append(containers[0].Env,
			corev1.EnvVar{Name: "METRICS_OUTPUT_FILE", Value: metricsFilePath(tj)})
	}

	volumes := []corev1.Volume{
		{
			Name: "checkpoints",
			VolumeSource: corev1.VolumeSource{
				PersistentVolumeClaim: &corev1.PersistentVolumeClaimVolumeSource{
					ClaimName: tj.Name + "-checkpoints",
				},
			},
		},
		{
			Name: "shm",
			VolumeSource: corev1.VolumeSource{
				EmptyDir: &corev1.EmptyDirVolumeSource{
					Medium:    corev1.StorageMediumMemory,
					SizeLimit: resourcePtr("64Gi"),
				},
			},
		},
	}

	if sidecarEnabled {
		volumes = append(volumes, corev1.Volume{
			Name: "sidecar-logs",
			VolumeSource: corev1.VolumeSource{
				EmptyDir: &corev1.EmptyDirVolumeSource{},
			},
		})
	}

	if mcEnabled {
		volumes = append(volumes, corev1.Volume{
			Name: metricsVolumeName,
			VolumeSource: corev1.VolumeSource{
				EmptyDir: &corev1.EmptyDirVolumeSource{},
			},
		})
	}

	annotations := map[string]string{
		"training.vsr.dev/model":     tj.Spec.Model,
		"training.vsr.dev/precision": tj.Spec.Precision,
	}
	if sidecarEnabled {
		annotations["prometheus.io/scrape"] = "true"
		annotations["prometheus.io/port"] = "9400"
		annotations["prometheus.io/path"] = "/metrics"
	}

	replicas := tj.Spec.NumNodes

	return &appsv1.StatefulSet{
		ObjectMeta: metav1.ObjectMeta{
			Name:      workerStsName(tj),
			Namespace: tj.Namespace,
			Labels:    labels,
		},
		Spec: appsv1.StatefulSetSpec{
			Replicas:    &replicas,
			ServiceName: headlessSvcName(tj),
			Selector: &metav1.LabelSelector{
				MatchLabels: labels,
			},
			// Parallel pod management: start all workers simultaneously
			// (rank 0 waits for others in torch.distributed.init_process_group)
			PodManagementPolicy: appsv1.ParallelPodManagement,
			Template: corev1.PodTemplateSpec{
				ObjectMeta: metav1.ObjectMeta{
					Labels:      labels,
					Annotations: annotations,
				},
				Spec: corev1.PodSpec{
					NodeSelector:                  tj.Spec.NodeSelector,
					RestartPolicy:                 corev1.RestartPolicyAlways,
					TerminationGracePeriodSeconds: int64Ptr(300),
					Tolerations: []corev1.Toleration{
						{
							Key:      "nvidia.com/gpu",
							Operator: corev1.TolerationOpExists,
							Effect:   corev1.TaintEffectNoSchedule,
						},
					},
					Containers: containers,
					Volumes:    volumes,
				},
			},
		},
	}
}

// buildGPUMonitorSidecar returns the gpu-monitor sidecar container.
// This was previously injected via a pod mutating webhook, but is now
// included directly in the StatefulSet pod template to avoid requiring
// a cluster-wide MutatingWebhookConfiguration on pods.
func buildGPUMonitorSidecar() corev1.Container {
	return corev1.Container{
		Name:    "gpu-monitor",
		Image:   "nvcr.io/nvidia/cloud-native/dcgm:3.3.6-1-ubuntu22.04",
		Command: []string{"bash", "-c"},
		Args:    []string{gpuMonitorScript()},
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
}

// gpuMonitorScript returns a minimal startup script for the GPU monitor.
// The full 5-subsystem observability agent script is defined in
// internal/webhook/trainjob_mutator.go (sidecarScript) and can be used
// here for the complete version. This is a simplified placeholder that
// starts the DCGM exporter and a basic health endpoint.
func gpuMonitorScript() string {
	return `#!/bin/bash
set -uo pipefail

LOG_DIR="${SIDECAR_LOG_DIR:-/var/log/gpu-monitor}"
mkdir -p "$LOG_DIR"
echo "$(date -u) gpu-monitor sidecar starting on $NODE_NAME ($POD_NAME)" | tee "$LOG_DIR/sidecar.log"

# Start DCGM exporter if available
if command -v dcgm-exporter &>/dev/null; then
    dcgm-exporter --address :9400 --collect-interval 5000 >> "$LOG_DIR/dcgm-exporter.log" 2>&1 &
    echo "$(date -u) DCGM exporter started (PID $!)" >> "$LOG_DIR/sidecar.log"
fi

# Health endpoint
while true; do
    echo -e "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n\r\n{\"status\":\"healthy\",\"node\":\"$NODE_NAME\"}" | \
        nc -l -p 9401 -q 1 2>/dev/null || sleep 1
done
`
}

func int64Ptr(v int64) *int64 { return &v }

func resourcePtr(s string) *resource.Quantity {
	q := resource.MustParse(s)
	return &q
}
