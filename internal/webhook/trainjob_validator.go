package webhook

import (
	"context"
	"fmt"
	"strings"
	"time"

	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"sigs.k8s.io/controller-runtime/pkg/webhook/admission"

	aiv1 "github.com/rootfs/trainjob-operator/api/v1alpha1"
)

//+kubebuilder:webhook:path=/validate-training-vsr-dev-v1alpha1-trainjob,mutating=false,failurePolicy=fail,sideEffects=None,groups=training.vsr.dev,resources=trainjobs,verbs=create;update,versions=v1alpha1,name=vtrainjob.training.vsr.dev,admissionReviewVersions=v1

// TrainJobValidator validates TrainJob create/update requests.
// This is the "first line of defense" that catches misconfigurations
// before they waste GPU time.
type TrainJobValidator struct{}

var _ admission.CustomValidator = &TrainJobValidator{}

func (v *TrainJobValidator) ValidateCreate(ctx context.Context, obj runtime.Object) (admission.Warnings, error) {
	tj := obj.(*aiv1.TrainJob)
	return v.validate(tj)
}

func (v *TrainJobValidator) ValidateUpdate(ctx context.Context, oldObj, newObj runtime.Object) (admission.Warnings, error) {
	tj := newObj.(*aiv1.TrainJob)
	oldTJ := oldObj.(*aiv1.TrainJob)

	warnings, err := v.validate(tj)

	// Prevent changing parallelism mid-training (would corrupt checkpoint)
	if oldTJ.Status.Phase == aiv1.PhaseRunning {
		if tj.Spec.TPDegree != oldTJ.Spec.TPDegree ||
			tj.Spec.PPDegree != oldTJ.Spec.PPDegree ||
			tj.Spec.NumNodes != oldTJ.Spec.NumNodes {
			return warnings, fmt.Errorf(
				"cannot change parallelism config (TP/PP/NumNodes) while job is Running; " +
					"stop the job first, then update")
		}
	}

	return warnings, err
}

func (v *TrainJobValidator) ValidateDelete(ctx context.Context, obj runtime.Object) (admission.Warnings, error) {
	return nil, nil
}

func (v *TrainJobValidator) validate(tj *aiv1.TrainJob) (admission.Warnings, error) {
	var errs field.ErrorList
	var warnings admission.Warnings

	specPath := field.NewPath("spec")

	// ── Rule 1: TP degree must not exceed GPUs per node ──
	// TP requires NVLink (intra-node), so TP degree > GPUs/node is physically impossible.
	if tj.Spec.TPDegree > tj.Spec.GPUsPerNode {
		errs = append(errs, field.Invalid(
			specPath.Child("tpDegree"),
			tj.Spec.TPDegree,
			fmt.Sprintf("TP degree (%d) cannot exceed GPUs per node (%d); "+
				"tensor parallelism requires NVLink within a single node",
				tj.Spec.TPDegree, tj.Spec.GPUsPerNode),
		))
	}

	// ── Rule 2: Total GPUs must be divisible by TP × PP × CP ──
	totalGPUs := tj.Spec.NumNodes * tj.Spec.GPUsPerNode
	cpDegree := tj.Spec.CPDegree
	if cpDegree == 0 {
		cpDegree = 1
	}
	parallelism := tj.Spec.TPDegree * tj.Spec.PPDegree * cpDegree
	if tj.Spec.TPDegree == 0 || tj.Spec.PPDegree == 0 {
		errs = append(errs, field.Invalid(
			specPath.Child("tpDegree"),
			parallelism,
			"TP and PP degree must both be > 0",
		))
	} else if totalGPUs%parallelism != 0 {
		errs = append(errs, field.Invalid(
			specPath,
			totalGPUs,
			fmt.Sprintf("total GPUs (%d = %d nodes × %d GPUs/node) must be divisible by TP×PP×CP (%d×%d×%d=%d); "+
				"remaining GPUs cannot form a complete FSDP group",
				totalGPUs, tj.Spec.NumNodes, tj.Spec.GPUsPerNode,
				tj.Spec.TPDegree, tj.Spec.PPDegree, cpDegree, parallelism),
		))
	}

	// ── Rule 3: FP8 only on Hopper+ GPUs ──
	if tj.Spec.Precision == "fp8" {
		gpuType := tj.Spec.NodeSelector["nvidia.com/gpu.product"]
		hopperOrNewer := strings.Contains(gpuType, "H100") ||
			strings.Contains(gpuType, "H200") ||
			strings.Contains(gpuType, "B100") ||
			strings.Contains(gpuType, "B200") ||
			strings.Contains(gpuType, "GB200")
		if !hopperOrNewer {
			errs = append(errs, field.Invalid(
				specPath.Child("precision"),
				"fp8",
				fmt.Sprintf("FP8 training requires Hopper+ GPU (H100/H200/B100/B200/GB200); "+
					"node selector specifies %q; use bf16 instead", gpuType),
			))
		}
	}

	// ── Rule 4: Checkpoint interval required for long jobs ──
	if tj.Spec.MaxRuntime != nil && tj.Spec.MaxRuntime.Duration > time.Hour {
		if !tj.Spec.Checkpoint.Enabled {
			errs = append(errs, field.Required(
				specPath.Child("checkpoint", "enabled"),
				"checkpointing must be enabled for jobs with maxRuntime > 1 hour; "+
					"without checkpoints, a failure at hour 23 loses all progress",
			))
		} else if tj.Spec.Checkpoint.IntervalMinutes == nil {
			errs = append(errs, field.Required(
				specPath.Child("checkpoint", "intervalMinutes"),
				"checkpoint interval required when checkpointing is enabled; "+
					"recommended: 30-60 minutes for large models",
			))
		}
	}

	// ── Rule 5: NumNodes must be > 0 ──
	if tj.Spec.NumNodes <= 0 {
		errs = append(errs, field.Invalid(
			specPath.Child("numNodes"),
			tj.Spec.NumNodes,
			"numNodes must be > 0",
		))
	}

	// ── Rule 6: GPUsPerNode must be 1, 2, 4, or 8 ──
	validGPUCounts := map[int32]bool{1: true, 2: true, 4: true, 8: true}
	if !validGPUCounts[tj.Spec.GPUsPerNode] {
		errs = append(errs, field.Invalid(
			specPath.Child("gpusPerNode"),
			tj.Spec.GPUsPerNode,
			"gpusPerNode must be 1, 2, 4, or 8 (matching physical GPU topology)",
		))
	}

	// ── Rule 7: Image must be specified ──
	if tj.Spec.Image == "" {
		errs = append(errs, field.Required(
			specPath.Child("image"),
			"training container image is required",
		))
	}

	// ── Rule 8: Checkpoint storage path required when enabled ──
	if tj.Spec.Checkpoint.Enabled && tj.Spec.Checkpoint.StoragePath == "" {
		errs = append(errs, field.Required(
			specPath.Child("checkpoint", "storagePath"),
			"storagePath required when checkpointing is enabled",
		))
	}

	// ── Rule 9: GPU memory estimation (if ModelSpec provided) ──
	if tj.Spec.ModelSpec != nil {
		memWarnings, memErrors := validateGPUMemory(tj)
		warnings = append(warnings, memWarnings...)
		errs = append(errs, memErrors...)
	}

	// ── Rule 10: NCCL bandwidth feasibility (if ModelSpec provided) ──
	if tj.Spec.ModelSpec != nil {
		bwWarnings, bwErrors := validateNCCLBandwidth(tj)
		warnings = append(warnings, bwWarnings...)
		errs = append(errs, bwErrors...)
	}

	// ── Warnings (non-blocking but informational) ──
	if tj.Spec.Precision == "fp32" {
		warnings = append(warnings, "fp32 precision is rarely used for LLM training; consider bf16 for better performance")
	}

	if tj.Spec.NumNodes > 64 && tj.Spec.SkipProlog {
		warnings = append(warnings, fmt.Sprintf(
			"skipProlog=true with %d nodes is risky; at this scale, ~%.0f%% chance of at least one unhealthy GPU",
			tj.Spec.NumNodes, 100*(1-pow(0.997, float64(tj.Spec.NumNodes*tj.Spec.GPUsPerNode)))))
	}

	if len(errs) > 0 {
		return warnings, fmt.Errorf("TrainJob validation failed: %s", errs.ToAggregate().Error())
	}
	return warnings, nil
}

// ════════════════════════════════════════════════════════════════
//  GPU Memory Estimation
// ════════════════════════════════════════════════════════════════

// gpuMemoryGB returns the usable GPU memory for a given GPU type.
// Usable = total - CUDA context - driver overhead (~5-8 GB).
func gpuMemoryGB(gpuType string) float64 {
	switch {
	case strings.Contains(gpuType, "H100-SXM"):
		return 72.0 // 80 GB - ~8 GB overhead
	case strings.Contains(gpuType, "H100-PCIe"):
		return 72.0
	case strings.Contains(gpuType, "H200"):
		return 133.0 // 141 GB - ~8 GB overhead
	case strings.Contains(gpuType, "A100-SXM4-80"):
		return 72.0
	case strings.Contains(gpuType, "A100-SXM4-40"), strings.Contains(gpuType, "A100-PCIe-40"):
		return 34.0
	case strings.Contains(gpuType, "L40"):
		return 42.0
	default:
		return 72.0 // conservative default
	}
}

func bytesPerParam(precision string) float64 {
	switch precision {
	case "fp32":
		return 4.0
	case "bf16":
		return 2.0
	case "fp8":
		return 1.0
	default:
		return 2.0
	}
}

// estimatePerGPUMemoryGB estimates peak GPU memory for training.
//
// Memory = Parameters + Gradients + Optimizer + Activations
//
// For a transformer with FSDP2 + TP + PP:
//   - Parameters are sharded across TP (intra-layer) and PP (inter-layer)
//     and FSDP (inter-node). FSDP2 only holds the shard, AllGathers on demand.
//   - Gradients: same shard size as parameters, always fp32 for accumulation.
//   - Optimizer (AdamW): 2 fp32 copies per param (momentum + variance), sharded by FSDP.
//   - Activations: proportional to seq_len × batch × hidden × layers / (TP × PP).
//     With activation checkpointing, reduced by ~70%.
//   - MHA attention scores: batch × heads_per_gpu × seq_len (O(seq_len) with FlashAttn).
func estimatePerGPUMemoryGB(tj *aiv1.TrainJob) (paramGB, gradGB, optGB, actGB, totalGB float64) {
	ms := tj.Spec.ModelSpec
	totalGPUs := float64(tj.Spec.NumNodes * tj.Spec.GPUsPerNode)
	tp := float64(tj.Spec.TPDegree)
	pp := float64(tj.Spec.PPDegree)
	cp := float64(tj.Spec.CPDegree)
	if cp == 0 {
		cp = 1
	}
	fsdp := totalGPUs / (tp * pp * cp)

	totalParams := ms.ParamsBillions * 1e9

	// Parameters: sharded across all parallelism dimensions
	// With FSDP2, each GPU holds params / (TP × PP × FSDP) during compute,
	// but AllGathers the full TP-local shard for each layer's forward/backward.
	// Peak param memory = params / PP (one stage) / FSDP (sharded) but
	// during AllGather, temporarily 2× for the gathering layer.
	paramsPerGPU := totalParams / (pp * fsdp)
	paramGB = paramsPerGPU * bytesPerParam(tj.Spec.Precision) / 1e9

	// Gradients: same shard as parameters, but always fp32 for ReduceScatter
	gradGB = paramsPerGPU * 4.0 / 1e9

	// Optimizer: AdamW has m (momentum) and v (variance), each fp32, sharded by FSDP
	// With FSDP2, optimizer states are fully sharded.
	optParamsPerGPU := totalParams / (tp * pp * fsdp)
	optGB = optParamsPerGPU * 8.0 / 1e9 // 4 bytes × 2 states

	// Activations: per-layer activation memory
	//   Per layer ≈ (seq_len/CP) × micro_batch × hidden × 10 bytes (empirical, includes
	//   attention output, FFN intermediates, layer norms, residuals)
	//   Divided by TP (attention heads split), PP (layers split), CP (sequence split)
	layersPerGPU := float64(ms.NumLayers) / pp
	seqPerCP := float64(ms.SeqLen) / cp
	activationPerLayer := seqPerCP * float64(ms.MicroBatchSize) * float64(ms.HiddenDim) * 10.0 / tp
	actTotal := activationPerLayer * layersPerGPU

	if ms.ActivationCheckpointing {
		actTotal *= 0.3 // ~70% reduction with selective recomputation
	}
	actGB = actTotal / 1e9

	// MHA-specific: attention score memory with FlashAttention
	// FlashAttn is O(seq_len) not O(seq_len^2), but still:
	// Working memory ≈ batch × heads_per_gpu × (seq_len/CP) × head_dim × 2 (Q·K^T intermediate)
	headsPerGPU := float64(ms.NumHeads) / tp
	headDim := float64(ms.HiddenDim) / float64(ms.NumHeads)
	mhaWorkingMem := float64(ms.MicroBatchSize) * headsPerGPU * seqPerCP * headDim * 2.0
	actGB += mhaWorkingMem / 1e9

	totalGB = paramGB + gradGB + optGB + actGB
	return
}

func validateGPUMemory(tj *aiv1.TrainJob) (warnings admission.Warnings, errs field.ErrorList) {
	gpuType := tj.Spec.NodeSelector["nvidia.com/gpu.product"]
	availableGB := gpuMemoryGB(gpuType)

	paramGB, gradGB, optGB, actGB, totalGB := estimatePerGPUMemoryGB(tj)

	specPath := field.NewPath("spec")

	if totalGB > availableGB {
		errs = append(errs, field.Invalid(
			specPath.Child("modelSpec"),
			fmt.Sprintf("%.1f GB", totalGB),
			fmt.Sprintf(
				"Estimated per-GPU memory (%.1f GB) exceeds available GPU memory (%.1f GB on %s). "+
					"Breakdown: params=%.1fGB, grads=%.1fGB, optimizer=%.1fGB, activations=%.1fGB. "+
					"Solutions: increase TP degree (current: %d), increase PP degree (current: %d), "+
					"enable activation checkpointing, reduce micro-batch size (current: %d), "+
					"or use a GPU with more memory",
				totalGB, availableGB, gpuType,
				paramGB, gradGB, optGB, actGB,
				tj.Spec.TPDegree, tj.Spec.PPDegree,
				tj.Spec.ModelSpec.MicroBatchSize),
		))
	} else if totalGB > availableGB*0.85 {
		// Warning at 85% — tight but might work
		warnings = append(warnings, fmt.Sprintf(
			"Estimated per-GPU memory %.1f GB is %.0f%% of available %.1f GB on %s; "+
				"this is tight and may OOM with larger batch sizes or sequence lengths. "+
				"Breakdown: params=%.1fGB, grads=%.1fGB, optimizer=%.1fGB, activations=%.1fGB",
			totalGB, (totalGB/availableGB)*100, availableGB, gpuType,
			paramGB, gradGB, optGB, actGB))
	}

	// Specific MHA check: attention heads must be evenly divisible by TP
	if tj.Spec.ModelSpec.NumHeads > 0 && int32(tj.Spec.TPDegree) > 0 {
		if tj.Spec.ModelSpec.NumHeads%int32(tj.Spec.TPDegree) != 0 {
			errs = append(errs, field.Invalid(
				specPath.Child("tpDegree"),
				tj.Spec.TPDegree,
				fmt.Sprintf(
					"NumHeads (%d) must be evenly divisible by TP degree (%d); "+
						"MHA splits attention heads across TP ranks — remainder heads can't be assigned",
					tj.Spec.ModelSpec.NumHeads, tj.Spec.TPDegree),
			))
		}
		// GQA check: KV heads must also be divisible by TP
		if tj.Spec.ModelSpec.NumKVHeads > 0 &&
			tj.Spec.ModelSpec.NumKVHeads%int32(tj.Spec.TPDegree) != 0 {
			errs = append(errs, field.Invalid(
				specPath.Child("tpDegree"),
				tj.Spec.TPDegree,
				fmt.Sprintf(
					"NumKVHeads (%d) must be evenly divisible by TP degree (%d); "+
						"GQA KV heads are split across TP ranks",
					tj.Spec.ModelSpec.NumKVHeads, tj.Spec.TPDegree),
			))
		}
	}

	return
}

// ════════════════════════════════════════════════════════════════
//  NCCL Bandwidth Feasibility
// ════════════════════════════════════════════════════════════════

// interconnectBandwidthGBps returns the per-GPU interconnect bandwidth.
func interconnectBandwidthGBps(gpuType string, intraNode bool) float64 {
	if intraNode {
		// NVLink bandwidth (bidirectional)
		switch {
		case strings.Contains(gpuType, "H100"):
			return 900.0 // NVLink 4.0: 900 GB/s
		case strings.Contains(gpuType, "H200"):
			return 900.0
		case strings.Contains(gpuType, "A100"):
			return 600.0 // NVLink 3.0: 600 GB/s
		default:
			return 300.0
		}
	}
	// Inter-node: InfiniBand or RoCE
	switch {
	case strings.Contains(gpuType, "H100"), strings.Contains(gpuType, "H200"):
		return 50.0 // 400 Gbps NDR IB = ~50 GB/s per GPU (with 4 HCAs)
	case strings.Contains(gpuType, "A100"):
		return 25.0 // 200 Gbps HDR IB = ~25 GB/s per GPU
	default:
		return 12.5
	}
}

func validateNCCLBandwidth(tj *aiv1.TrainJob) (warnings admission.Warnings, errs field.ErrorList) {
	ms := tj.Spec.ModelSpec
	gpuType := tj.Spec.NodeSelector["nvidia.com/gpu.product"]
	specPath := field.NewPath("spec")

	tp := float64(tj.Spec.TPDegree)
	pp := float64(tj.Spec.PPDegree)
	cpd := float64(tj.Spec.CPDegree)
	if cpd == 0 {
		cpd = 1
	}
	totalGPUs := float64(tj.Spec.NumNodes * tj.Spec.GPUsPerNode)
	fsdp := totalGPUs / (tp * pp * cpd)

	nvlinkBW := interconnectBandwidthGBps(gpuType, true)
	ibBW := interconnectBandwidthGBps(gpuType, false)

	// ── TP bandwidth check ──
	// TP does AllReduce on every layer's output: 2 × hidden × seq × batch × bytes
	// "2×" because AllReduce = ReduceScatter + AllGather
	// This happens for EVERY layer, so total per step = 2 × volume × num_layers / PP
	if tp > 1 {
		tpVolumePerLayer := 2.0 * float64(ms.HiddenDim) * float64(ms.SeqLen) *
			float64(ms.MicroBatchSize) * bytesPerParam(tj.Spec.Precision)
		layersPerStage := float64(ms.NumLayers) / pp
		tpVolumePerStep := tpVolumePerLayer * layersPerStage

		// Assume compute time per step ≈ 100ms for a large model (conservative).
		// TP communication must complete within this window.
		tpBandwidthNeeded := tpVolumePerStep / 0.1 / 1e9 // GB/s needed

		if tpBandwidthNeeded > nvlinkBW*0.8 {
			warnings = append(warnings, fmt.Sprintf(
				"TP AllReduce needs ~%.0f GB/s bandwidth (%.1f GB per step), "+
					"NVLink provides %.0f GB/s; TP communication may become the bottleneck. "+
					"Consider reducing micro-batch size (%d) or sequence length (%d)",
				tpBandwidthNeeded, tpVolumePerStep/1e9,
				nvlinkBW, ms.MicroBatchSize, ms.SeqLen))
		}
	}

	// ── FSDP bandwidth check ──
	// FSDP does AllGather (before forward) + ReduceScatter (after backward) per layer.
	// Volume per layer = 2 × params_per_layer × bytes
	// This traverses inter-node IB if FSDP degree > GPUs per node.
	if fsdp > 1 && fsdp > float64(tj.Spec.GPUsPerNode) {
		paramsPerLayer := ms.ParamsBillions * 1e9 / float64(ms.NumLayers)
		fsdpVolumePerLayer := 2.0 * paramsPerLayer * bytesPerParam(tj.Spec.Precision)
		layersPerStage := float64(ms.NumLayers) / pp
		fsdpVolumePerStep := fsdpVolumePerLayer * layersPerStage

		fsdpBandwidthNeeded := fsdpVolumePerStep / 0.1 / 1e9

		if fsdpBandwidthNeeded > ibBW*0.7 {
			warnings = append(warnings, fmt.Sprintf(
				"FSDP AllGather+ReduceScatter needs ~%.0f GB/s bandwidth (%.1f GB per step), "+
					"InfiniBand provides ~%.0f GB/s per GPU; FSDP communication may bottleneck. "+
					"Consider increasing TP degree to keep more computation intra-node, "+
					"or reducing model size per FSDP shard",
				fsdpBandwidthNeeded, fsdpVolumePerStep/1e9, ibBW))
		}
	}

	// ── PP communication check ──
	// PP sends activations point-to-point between stages.
	// Volume = micro_batch × seq_len × hidden × bytes (once forward, once backward = 2×)
	if pp > 1 {
		ppVolume := 2.0 * float64(ms.MicroBatchSize) * float64(ms.SeqLen) *
			float64(ms.HiddenDim) * bytesPerParam(tj.Spec.Precision)

		ppNeedsInterNode := pp > float64(tj.Spec.GPUsPerNode)
		var ppBW float64
		if ppNeedsInterNode {
			ppBW = ibBW
		} else {
			ppBW = nvlinkBW
		}

		ppBandwidthNeeded := ppVolume / 0.1 / 1e9
		if ppBandwidthNeeded > ppBW*0.5 {
			warnings = append(warnings, fmt.Sprintf(
				"PP activation transfer needs ~%.0f GB/s (%.1f GB per step); "+
					"consider pipeline micro-batching to overlap compute and communication",
				ppBandwidthNeeded, ppVolume/1e9))
		}
	}

	// ── Compute/Communication ratio warning ──
	// If FSDP communication volume > 50% of total compute time, training is comm-bound.
	if fsdp > float64(tj.Spec.GPUsPerNode) && ms.ParamsBillions > 10 {
		commRatio := ibBW / (ibBW + nvlinkBW)
		if commRatio > 0.4 {
			errs = append(errs, field.Invalid(
				specPath.Child("numNodes"),
				tj.Spec.NumNodes,
				fmt.Sprintf(
					"With %.0fB params, FSDP degree %.0f, and TP=%d, the job is likely "+
						"communication-bound (inter-node BW %.0f GB/s vs intra-node %.0f GB/s). "+
						"Consider: (1) increase TP to %d to keep more work intra-node, "+
						"(2) increase PP to reduce FSDP degree, "+
						"(3) use fewer nodes with more GPUs per node",
					ms.ParamsBillions, fsdp, tj.Spec.TPDegree,
					ibBW, nvlinkBW,
					tj.Spec.GPUsPerNode),
			))
		}
	}

	return
}

func pow(base, exp float64) float64 {
	result := 1.0
	for i := 0; i < int(exp); i++ {
		result *= base
	}
	return result
}
