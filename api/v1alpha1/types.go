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

	// MetricsConfig controls how training metrics are collected and exposed.
	// When enabled, a metrics-collector sidecar reads a shared metrics file from the
	// training container and patches the TrainJob status sub-resource periodically.
	// +optional
	MetricsConfig *MetricsConfig `json:"metricsConfig,omitempty"`

	// EvalConfig specifies post-training evaluation. When set, the reconciler
	// creates an eval Job after training succeeds and writes results to status.eval.
	// +optional
	EvalConfig *EvalConfig `json:"evalConfig,omitempty"`
}

// MetricsConfig controls how training metrics are collected.
type MetricsConfig struct {
	// Enabled controls whether the metrics-collector sidecar is added.
	// Defaults to true when the training image writes to the metrics file path.
	Enabled bool `json:"enabled"`

	// MetricsFilePath is the path inside the training container where metrics JSON
	// is written. The metrics-collector sidecar reads this file and patches the
	// TrainJob status. Defaults to /var/run/training/metrics.json.
	// +optional
	MetricsFilePath string `json:"metricsFilePath,omitempty"`

	// ScrapeIntervalSeconds is how often the sidecar reads the metrics file.
	// Defaults to 30.
	// +optional
	ScrapeIntervalSeconds *int32 `json:"scrapeIntervalSeconds,omitempty"`

	// PrometheusPort is the port for Prometheus-format metrics export from the
	// metrics-collector sidecar. Defaults to 9402.
	// +optional
	PrometheusPort *int32 `json:"prometheusPort,omitempty"`
}

// EvalConfig specifies post-training evaluation.
type EvalConfig struct {
	// Image is the container image for evaluation. If empty, uses the training image.
	// +optional
	Image string `json:"image,omitempty"`

	// Command for the eval container.
	// +optional
	Command []string `json:"command,omitempty"`

	// Args for the eval container.
	// +optional
	Args []string `json:"args,omitempty"`

	// DatasetPath is the path to the evaluation dataset (PVC mount or S3 URI).
	// +optional
	DatasetPath string `json:"datasetPath,omitempty"`

	// PreviousModelPath is the path to the previous model for regression comparison.
	// +optional
	PreviousModelPath string `json:"previousModelPath,omitempty"`

	// GPUsPerNode for the eval job (typically less than training).
	// +optional
	GPUsPerNode *int32 `json:"gpusPerNode,omitempty"`
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

	// TotalSteps is the expected total iterations (if known from the training script).
	// +optional
	TotalSteps int64 `json:"totalSteps,omitempty"`

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

	// Training captures live metrics from the training process.
	// Updated periodically by the metrics-collector sidecar via the status sub-resource.
	// +optional
	Training *TrainingMetrics `json:"training,omitempty"`

	// Eval holds post-training evaluation results.
	// Written by the eval job or the model-eval agent after training completes.
	// +optional
	Eval *EvalMetrics `json:"eval,omitempty"`

	// Deployment captures metrics from the model conversion and loading phase.
	// Written by the model-install agent or a post-training job.
	// +optional
	Deployment *DeploymentMetrics `json:"deployment,omitempty"`

	// Serving captures live production metrics after the model is deployed.
	// Updated by the serving observer or scraped from vLLM Prometheus metrics.
	// +optional
	Serving *ServingMetrics `json:"serving,omitempty"`

	// CheckpointStatus provides detailed info about the checkpoint subsystem.
	// +optional
	CheckpointStatus *CheckpointMetrics `json:"checkpointStatus,omitempty"`
}

// TrainingMetrics captures real-time signals from the training loop.
// The metrics-collector sidecar reads these from a shared metrics file
// written by the training script and patches the TrainJob status.
type TrainingMetrics struct {
	// Loss
	TrainLoss     *float64 `json:"trainLoss,omitempty"`
	ValLoss       *float64 `json:"valLoss,omitempty"`
	LossDeltaPerK *float64 `json:"lossDeltaPerK,omitempty"`

	// Gradient health
	GradientNorm    *float64 `json:"gradientNorm,omitempty"`
	GradientNormMax *float64 `json:"gradientNormMax,omitempty"`
	ZeroGradientPct *float64 `json:"zeroGradientPct,omitempty"`

	// Throughput
	TokensPerSecond  *float64 `json:"tokensPerSecond,omitempty"`
	SamplesPerSecond *float64 `json:"samplesPerSecond,omitempty"`
	MFU              *float64 `json:"mfu,omitempty"`

	// Communication efficiency (fraction of step time spent in NCCL collectives)
	CommComputeRatio *float64 `json:"commComputeRatio,omitempty"`
	StepTimeSec      *float64 `json:"stepTimeSec,omitempty"`

	// Memory
	PeakMemoryGB *float64 `json:"peakMemoryGB,omitempty"`
	AllocatedGB  *float64 `json:"allocatedGB,omitempty"`
	OOMEvents    int32    `json:"oomEvents,omitempty"`

	// Straggler detection across workers
	StepTimeP50Sec *float64 `json:"stepTimeP50Sec,omitempty"`
	StepTimeP99Sec *float64 `json:"stepTimeP99Sec,omitempty"`
	StragglerRatio *float64 `json:"stragglerRatio,omitempty"`

	// Hardware events
	ECCErrors     int32 `json:"eccErrors,omitempty"`
	ThermalEvents int32 `json:"thermalEvents,omitempty"`
	NVLinkErrors  int32 `json:"nvlinkErrors,omitempty"`

	// Timestamp of last metrics update
	LastUpdated *metav1.Time `json:"lastUpdated,omitempty"`
}

// EvalMetrics holds post-training evaluation results.
type EvalMetrics struct {
	// Benchmarks is a list of task-specific evaluation results.
	Benchmarks []BenchmarkResult `json:"benchmarks,omitempty"`

	// QuantizationSensitivity shows how eval metrics degrade under quantization.
	// Key is precision (fp16, int8, fp8), value is the primary benchmark score.
	// +optional
	QuantizationSensitivity map[string]float64 `json:"quantizationSensitivity,omitempty"`

	// InferenceLatency at eval time (single-request, not under load).
	InferenceLatencyP50Ms *float64 `json:"inferenceLatencyP50Ms,omitempty"`
	InferenceLatencyP99Ms *float64 `json:"inferenceLatencyP99Ms,omitempty"`

	// EmbeddingQuality metrics for embedding models.
	// +optional
	EmbeddingQuality *EmbeddingQualityMetrics `json:"embeddingQuality,omitempty"`

	// EvalTime is when the evaluation was completed.
	// +optional
	EvalTime *metav1.Time `json:"evalTime,omitempty"`

	// Verdict is the eval agent's recommendation: "promote", "rollback", "retrain".
	// +optional
	Verdict string `json:"verdict,omitempty"`
}

// BenchmarkResult captures a single evaluation metric compared to the previous model.
type BenchmarkResult struct {
	Name          string   `json:"name"`
	Value         float64  `json:"value"`
	PreviousValue *float64 `json:"previousValue,omitempty"`
	Delta         *float64 `json:"delta,omitempty"`
	Threshold     *float64 `json:"threshold,omitempty"`
	Passed        *bool    `json:"passed,omitempty"`
}

// EmbeddingQualityMetrics captures embedding-space health for retrieval models.
type EmbeddingQualityMetrics struct {
	RetrievalRecallAt10 *float64 `json:"retrievalRecallAt10,omitempty"`
	NDCG                *float64 `json:"ndcg,omitempty"`
	MRR                 *float64 `json:"mrr,omitempty"`
	InterClassDistance  *float64 `json:"interClassDistance,omitempty"`
	IntraClassVariance  *float64 `json:"intraClassVariance,omitempty"`
}

// DeploymentMetrics captures signals from the checkpoint→serving conversion pipeline.
type DeploymentMetrics struct {
	// Conversion
	ConversionTimeSec *float64 `json:"conversionTimeSec,omitempty"`
	OutputSizeGB      *float64 `json:"outputSizeGB,omitempty"`
	MaxAbsError       *float64 `json:"maxAbsError,omitempty"`
	OutputFormat      string   `json:"outputFormat,omitempty"`

	// Loading
	LoadTimeSec        *float64 `json:"loadTimeSec,omitempty"`
	ServingMemoryGB    *float64 `json:"servingMemoryGB,omitempty"`
	TimeToFirstTokenMs *float64 `json:"timeToFirstTokenMs,omitempty"`

	// Smoke test
	SmokeTestPassed *bool  `json:"smokeTestPassed,omitempty"`
	SmokeTestError  string `json:"smokeTestError,omitempty"`

	// Compatibility
	TestedServingTP    int32  `json:"testedServingTP,omitempty"`
	VLLMVersion        string `json:"vllmVersion,omitempty"`
	CompatibilityNotes string `json:"compatibilityNotes,omitempty"`

	// Timestamp
	DeployedAt *metav1.Time `json:"deployedAt,omitempty"`
}

// ServingMetrics captures production-time signals after the model is live.
// These flow backwards to the agent system to close the improvement loop.
type ServingMetrics struct {
	// Routing accuracy (semantic router specific)
	RoutingAccuracy    *float64 `json:"routingAccuracy,omitempty"`
	RoutingAccuracyP7d *float64 `json:"routingAccuracyP7d,omitempty"`

	// Embedding drift: statistical distance from training distribution.
	// A rising value means the model is seeing OOD inputs.
	EmbeddingDriftKL  *float64 `json:"embeddingDriftKL,omitempty"`
	EmbeddingDriftMMD *float64 `json:"embeddingDriftMMD,omitempty"`

	// Latency under production load
	LatencyP50Ms *float64 `json:"latencyP50Ms,omitempty"`
	LatencyP95Ms *float64 `json:"latencyP95Ms,omitempty"`
	LatencyP99Ms *float64 `json:"latencyP99Ms,omitempty"`

	// Throughput
	RequestsPerSecond *float64 `json:"requestsPerSecond,omitempty"`
	QueueDepth        *float64 `json:"queueDepth,omitempty"`

	// Cache effectiveness
	CacheHitRate *float64 `json:"cacheHitRate,omitempty"`

	// A/B comparison (when running alongside a previous model version)
	ABTestActive    bool     `json:"abTestActive,omitempty"`
	ABPrimaryMetric string   `json:"abPrimaryMetric,omitempty"`
	ABCurrentValue  *float64 `json:"abCurrentValue,omitempty"`
	ABBaselineValue *float64 `json:"abBaselineValue,omitempty"`
	ABPValue        *float64 `json:"abPValue,omitempty"`

	// User-level signal (downstream success/reformulation rate)
	ReformulationRate *float64 `json:"reformulationRate,omitempty"`

	// Timestamp
	LastScraped *metav1.Time `json:"lastScraped,omitempty"`
}

// CheckpointMetrics provides detailed info about the checkpoint subsystem.
type CheckpointMetrics struct {
	LastSaved         *metav1.Time `json:"lastSaved,omitempty"`
	SizeGB            *float64     `json:"sizeGB,omitempty"`
	SaveDurationSec   *float64     `json:"saveDurationSec,omitempty"`
	Validated         *bool        `json:"validated,omitempty"`
	TotalSaved        int32        `json:"totalSaved,omitempty"`
	TotalValidated    int32        `json:"totalValidated,omitempty"`
	FailedValidations int32        `json:"failedValidations,omitempty"`
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
	PhaseEvaluating    TrainJobPhase = "Evaluating"
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
