package v1alpha1

import (
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// TrainJobSpec defines the desired state of a distributed GPU training job.
type TrainJobSpec struct {
	// Model identifies the model being trained (e.g., "llama-3-70b").
	Model string `json:"model"`

	// Image is the training container image.
	Image string `json:"image"`

	// NumNodes is the number of worker nodes (each gets GPUsPerNode GPUs).
	NumNodes int32 `json:"numNodes"`

	// GPUsPerNode is the number of GPUs requested per node (typically 8 for DGX).
	GPUsPerNode int32 `json:"gpusPerNode"`

	// Parallelism configuration for 4D parallelism.
	TPDegree int32 `json:"tpDegree"`
	PPDegree int32 `json:"ppDegree"`
	// FSDP degree is inferred: totalGPUs / (TP * PP * CP)

	// CPDegree is the context-parallelism degree for long sequences (Ring Attention).
	// Defaults to 1. Only useful for SeqLen >= 32k.
	// +optional
	CPDegree int32 `json:"cpDegree,omitempty"`

	// Precision is the training precision: "bf16", "fp8", "fp32".
	Precision string `json:"precision"`

	// AutoParallelism enables the auto-configuration advisor in the mutating webhook.
	// When true, the webhook derives optimal TP/PP/FSDP/CP/micro-batch/precision from
	// the model architecture (ModelSpec or model name in ModelRegistry) and the cluster
	// hardware. User-specified parallelism values are ignored and overwritten.
	// +optional
	AutoParallelism bool `json:"autoParallelism,omitempty"`

	// ModelSpec describes the model architecture for memory/bandwidth estimation.
	// If provided, the webhook validates that the model fits in GPU memory and
	// that the parallelism config has sufficient communication bandwidth.
	// +optional
	ModelSpec *ModelArchSpec `json:"modelSpec,omitempty"`

	// NodeSelector constrains which nodes can run the training pods.
	// +optional
	NodeSelector map[string]string `json:"nodeSelector,omitempty"`

	// Checkpoint configuration.
	Checkpoint CheckpointSpec `json:"checkpoint"`

	// MaxRuntime is the maximum duration before the job is killed.
	// +optional
	MaxRuntime *metav1.Duration `json:"maxRuntime,omitempty"`

	// Command overrides the container entrypoint.
	// +optional
	Command []string `json:"command,omitempty"`

	// Args are additional arguments to the training script.
	// +optional
	Args []string `json:"args,omitempty"`

	// Env are additional environment variables injected into worker containers.
	// +optional
	Env []EnvVar `json:"env,omitempty"`

	// Resources per worker container (cpu, memory beyond GPUs).
	// +optional
	CPUPerNode resource.Quantity `json:"cpuPerNode,omitempty"`
	MemPerNode resource.Quantity `json:"memPerNode,omitempty"`

	// SkipProlog disables the pre-job GPU health check. Use only for short test jobs.
	// +optional
	SkipProlog bool `json:"skipProlog,omitempty"`

	// EnableSidecar enables the GPU monitoring sidecar in the worker StatefulSet.
	// Defaults to true.
	// +optional
	EnableSidecar *bool `json:"enableSidecar,omitempty"`

	// Suspend indicates that the TrainJob should not create child resources.
	// When set to true by an external admission controller (e.g., Kueue), the
	// reconciler will not create the prolog Job, headless Service, or worker
	// StatefulSet. When Kueue admits the workload, it sets this to false
	// and the reconciler proceeds normally.
	// +optional
	Suspend *bool `json:"suspend,omitempty"`
}

type EnvVar struct {
	Name  string `json:"name"`
	Value string `json:"value"`
}

// ModelArchSpec describes transformer model architecture for memory estimation.
type ModelArchSpec struct {
	// ParamsBillions is the total parameter count in billions (e.g., 70 for Llama-3-70B).
	ParamsBillions float64 `json:"paramsBillions"`

	// HiddenDim is the hidden dimension size (e.g., 8192 for Llama-3-70B).
	HiddenDim int64 `json:"hiddenDim"`

	// NumLayers is the number of transformer layers (e.g., 80 for Llama-3-70B).
	NumLayers int32 `json:"numLayers"`

	// NumHeads is the total number of attention heads (e.g., 64 for Llama-3-70B).
	NumHeads int32 `json:"numHeads"`

	// NumKVHeads is the number of KV heads for GQA (e.g., 8 for Llama-3-70B).
	// If 0 or equal to NumHeads, assumes standard MHA.
	// +optional
	NumKVHeads int32 `json:"numKVHeads,omitempty"`

	// SeqLen is the max sequence length (e.g., 8192).
	SeqLen int64 `json:"seqLen"`

	// MicroBatchSize is the per-GPU micro-batch size.
	MicroBatchSize int32 `json:"microBatchSize"`

	// ActivationCheckpointing enables selective activation recomputation.
	// Reduces activation memory by ~70% at cost of ~33% more compute.
	// +optional
	ActivationCheckpointing bool `json:"activationCheckpointing,omitempty"`
}

type CheckpointSpec struct {
	// Enabled controls whether periodic checkpointing is active.
	Enabled bool `json:"enabled"`

	// IntervalMinutes is how often to checkpoint (in minutes).
	// +optional
	IntervalMinutes *int32 `json:"intervalMinutes,omitempty"`

	// StoragePath is the PVC mount path or S3 URI for checkpoint storage.
	StoragePath string `json:"storagePath"`

	// RetainCount is the number of checkpoints to keep (FIFO deletion of older ones).
	// +optional
	RetainCount *int32 `json:"retainCount,omitempty"`

	// ValidateOnSave runs a validation job after each checkpoint save.
	// +optional
	ValidateOnSave bool `json:"validateOnSave,omitempty"`
}

// TrainJobStatus defines the observed state of the TrainJob.
type TrainJobStatus struct {
	// Phase is the high-level state: Pending, PrologRunning, PrologPassed,
	// PrologFailed, Running, Checkpointing, Succeeded, Failed.
	Phase TrainJobPhase `json:"phase,omitempty"`

	// Conditions provide detailed status signals following K8s conventions.
	// +optional
	Conditions []metav1.Condition `json:"conditions,omitempty"`

	// StartTime is when the training workers started.
	// +optional
	StartTime *metav1.Time `json:"startTime,omitempty"`

	// CompletionTime is when the job finished.
	// +optional
	CompletionTime *metav1.Time `json:"completionTime,omitempty"`

	// CurrentStep is the training iteration count reported by workers.
	CurrentStep int64 `json:"currentStep,omitempty"`

	// LastCheckpoint is the path to the most recent valid checkpoint.
	// +optional
	LastCheckpoint string `json:"lastCheckpoint,omitempty"`

	// LastCheckpointTime is when the last checkpoint was saved.
	// +optional
	LastCheckpointTime *metav1.Time `json:"lastCheckpointTime,omitempty"`

	// HealthyNodes is the count of nodes that passed prolog.
	HealthyNodes int32 `json:"healthyNodes,omitempty"`

	// TotalNodes is the count of nodes allocated.
	TotalNodes int32 `json:"totalNodes,omitempty"`

	// ReadyWorkers is the count of worker pods in Ready state.
	ReadyWorkers int32 `json:"readyWorkers,omitempty"`

	// FailureReason is a machine-readable reason for failure.
	// +optional
	FailureReason string `json:"failureReason,omitempty"`

	// FailureMessage is a human-readable description of the failure.
	// +optional
	FailureMessage string `json:"failureMessage,omitempty"`
}

type TrainJobPhase string

const (
	PhaseSuspended     TrainJobPhase = "Suspended"
	PhasePending       TrainJobPhase = "Pending"
	PhasePrologRunning TrainJobPhase = "PrologRunning"
	PhasePrologPassed  TrainJobPhase = "PrologPassed"
	PhasePrologFailed  TrainJobPhase = "PrologFailed"
	PhaseRunning       TrainJobPhase = "Running"
	PhaseCheckpointing TrainJobPhase = "Checkpointing"
	PhaseSucceeded     TrainJobPhase = "Succeeded"
	PhaseFailed        TrainJobPhase = "Failed"
)

//+kubebuilder:object:root=true
//+kubebuilder:subresource:status
//+kubebuilder:printcolumn:name="Phase",type="string",JSONPath=".status.phase"
//+kubebuilder:printcolumn:name="Nodes",type="integer",JSONPath=".spec.numNodes"
//+kubebuilder:printcolumn:name="GPUs",type="string",JSONPath=".spec.gpusPerNode",priority=1
//+kubebuilder:printcolumn:name="Model",type="string",JSONPath=".spec.model"
//+kubebuilder:printcolumn:name="Step",type="integer",JSONPath=".status.currentStep"
//+kubebuilder:printcolumn:name="Age",type="date",JSONPath=".metadata.creationTimestamp"

// TrainJob represents a distributed GPU training job with health checks,
// checkpoint management, and automatic fault recovery.
type TrainJob struct {
	metav1.TypeMeta   `json:",inline"`
	metav1.ObjectMeta `json:"metadata,omitempty"`

	Spec   TrainJobSpec   `json:"spec,omitempty"`
	Status TrainJobStatus `json:"status,omitempty"`
}

//+kubebuilder:object:root=true

// TrainJobList contains a list of TrainJobs.
type TrainJobList struct {
	metav1.TypeMeta `json:",inline"`
	metav1.ListMeta `json:"metadata,omitempty"`
	Items           []TrainJob `json:"items"`
}

func init() {
	SchemeBuilder.Register(&TrainJob{}, &TrainJobList{})
}
