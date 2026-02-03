package webhook

import (
	"fmt"
	"math"
	"sort"

	aiv1 "github.com/rootfs/trainjob-operator/api/v1alpha1"
)

// ParallelismConfig holds the output of the auto-configurator.
type ParallelismConfig struct {
	TP                      int32
	PP                      int32
	FSDP                    int32 // derived
	CP                      int32
	MicroBatchSize          int32
	ActivationCheckpointing bool
	Precision               string
	Score                   float64 // estimated throughput (higher is better)
	Reason                  string  // human-readable explanation of why this config was chosen
}

// AutoConfigureParallelism finds the optimal TP/PP/FSDP/CP/batch/precision
// configuration for a given model on a given cluster.
//
// The search follows the hardware hierarchy for early pruning:
//
//	Level 0 (TP)  — NVLink tier, intra-node. Decided first because it's the
//	                tightest constraint: must be power-of-2, divide attention
//	                heads, and fit within a single node's GPUs.
//	Level 1 (PP)  — IB fabric tier, inter-node pipeline. Decided second because
//	                it determines per-stage layer count and thus base memory.
//	Level 2 (CP)  — IB fabric tier, inter-node ring attention. Only relevant
//	                for seq_len >= 32k; usually 1.
//	Level 3 (FSDP)— Derived: totalGPUs / (TP × PP × CP). Not a search variable.
//	Leaf  (precision, micro-batch, act-ckpt) — explored only for (TP, PP, CP)
//	                combos that pass the memory floor check.
//
// At each level, a lower-bound memory estimate prunes impossible branches
// before descending. For example, if a 70B model with TP=2 can't fit even
// with max PP, act_ckpt=true, and micro_batch=1, the entire TP=2 subtree
// is skipped without evaluating any leaf configs.
//
// The total search space is small (~1,920 upper bound) so brute-force is
// already fast, but hierarchical pruning cuts it further for large models
// where most low-TP configs are infeasible.
func AutoConfigureParallelism(
	model aiv1.ModelArchSpec,
	gpu GPUSpec,
	gpusPerNode int32,
	numNodes int32,
) ParallelismConfig {
	totalGPUs := numNodes * gpusPerNode
	var candidates []ParallelismConfig
	explored := 0
	pruned := 0

	tpCandidates := validTPDegrees(model, gpusPerNode)
	ppCandidates := validPPDegrees(model, numNodes)
	cpCandidates := []int32{1}
	if model.SeqLen >= 32768 {
		cpCandidates = append(cpCandidates, 2, 4)
	}
	mbCandidates := []int32{1, 2, 4, 8}
	precisionCandidates := []string{"bf16"}
	if gpu.SupportsFP8 {
		precisionCandidates = append(precisionCandidates, "fp8")
	}

	// Level 0: TP (NVLink tier)
	for _, tp := range tpCandidates {
		// Lower-bound memory check for this TP: use max PP (most sharding),
		// act_ckpt=true (least activation memory), micro_batch=1, CP=max.
		maxPP := ppCandidates[len(ppCandidates)-1]
		maxCP := cpCandidates[len(cpCandidates)-1]
		bestCaseFSDP := totalGPUs / (tp * maxPP * maxCP)
		if bestCaseFSDP < 1 {
			bestCaseFSDP = 1
		}
		floorMem := estimateMemoryGB(model, gpu, tp, maxPP, bestCaseFSDP, maxCP, 1, true, precisionCandidates[len(precisionCandidates)-1])
		if floorMem > gpu.MemoryGB {
			pruned++
			continue // entire TP subtree is infeasible
		}

		// Level 1: PP (IB pipeline tier)
		for _, pp := range ppCandidates {
			maxCPForPP := cpCandidates[len(cpCandidates)-1]
			fsdpFloor := totalGPUs / (tp * pp * maxCPForPP)
			if fsdpFloor < 1 {
				fsdpFloor = 1
			}
			ppFloorMem := estimateMemoryGB(model, gpu, tp, pp, fsdpFloor, maxCPForPP, 1, true, precisionCandidates[len(precisionCandidates)-1])
			if ppFloorMem > gpu.MemoryGB {
				pruned++
				continue // all configs at this (TP, PP) are infeasible
			}

			// Level 2: CP (ring attention tier)
			for _, cp := range cpCandidates {
				fsdp := totalGPUs / (tp * pp * cp)
				if fsdp < 1 || tp*pp*cp*fsdp != totalGPUs {
					continue
				}

				// Leaf: precision × micro-batch × activation checkpointing
				for _, prec := range precisionCandidates {
					for _, mb := range mbCandidates {
						for _, actCkpt := range []bool{false, true} {
							explored++
							mem := estimateMemoryGB(model, gpu, tp, pp, fsdp, cp, mb, actCkpt, prec)
							if mem > gpu.MemoryGB*0.95 {
								continue
							}

							score := estimateThroughput(model, gpu, tp, pp, fsdp, cp, mb, actCkpt, prec, gpusPerNode)
							candidates = append(candidates, ParallelismConfig{
								TP:                      tp,
								PP:                      pp,
								FSDP:                    fsdp,
								CP:                      cp,
								MicroBatchSize:          mb,
								ActivationCheckpointing: actCkpt,
								Precision:               prec,
								Score:                   score,
							})
						}
					}
				}
			}
		}
	}

	if len(candidates) == 0 {
		return ParallelismConfig{
			Reason: "no feasible configuration found; model may be too large for this cluster",
		}
	}

	sort.Slice(candidates, func(i, j int) bool {
		return candidates[i].Score > candidates[j].Score
	})

	best := candidates[0]
	bestMem := estimateMemoryGB(model, gpu, best.TP, best.PP, best.FSDP, best.CP,
		best.MicroBatchSize, best.ActivationCheckpointing, best.Precision)
	best.Reason = fmt.Sprintf(
		"auto-configured: TP=%d (heads=%d÷%d), PP=%d (layers=%d÷%d), "+
			"FSDP=%d, CP=%d, micro_batch=%d, act_ckpt=%v, precision=%s; "+
			"estimated memory=%.1fGB/%.1fGB (%.0f%% utilization); "+
			"explored %d configs, pruned %d branches, scored %.1f (highest throughput)",
		best.TP, model.NumHeads, best.TP,
		best.PP, model.NumLayers, best.PP,
		best.FSDP, best.CP, best.MicroBatchSize,
		best.ActivationCheckpointing, best.Precision,
		bestMem, gpu.MemoryGB, bestMem/gpu.MemoryGB*100,
		explored, pruned, best.Score,
	)

	return best
}

// ════════════════════════════════════════════════════════════════
//  Search space generation
// ════════════════════════════════════════════════════════════════

// validTPDegrees returns TP values that are:
//   - powers of 2 (NVLink topology)
//   - ≤ GPUs per node (must be intra-node)
//   - evenly divide num_heads AND num_kv_heads
func validTPDegrees(model aiv1.ModelArchSpec, gpusPerNode int32) []int32 {
	var valid []int32
	for tp := int32(1); tp <= gpusPerNode; tp *= 2 {
		if model.NumHeads%tp != 0 {
			continue
		}
		if model.NumKVHeads > 0 && model.NumKVHeads%tp != 0 {
			continue
		}
		valid = append(valid, tp)
	}
	return valid
}

// validPPDegrees returns PP values that evenly divide num_layers.
func validPPDegrees(model aiv1.ModelArchSpec, maxNodes int32) []int32 {
	var valid []int32
	for pp := int32(1); pp <= maxNodes && pp <= model.NumLayers; pp++ {
		if model.NumLayers%pp == 0 {
			valid = append(valid, pp)
		}
	}
	return valid
}

// ════════════════════════════════════════════════════════════════
//  Memory estimation
// ════════════════════════════════════════════════════════════════

func bytesPerDtype(precision string) float64 {
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

func estimateMemoryGB(
	model aiv1.ModelArchSpec,
	gpu GPUSpec,
	tp, pp, fsdp, cp, mb int32,
	actCkpt bool,
	precision string,
) float64 {
	totalParams := model.ParamsBillions * 1e9
	bpd := bytesPerDtype(precision)

	// Parameters: sharded by PP (layers split) and FSDP (AllGather on demand)
	// Peak = one full stage's params during AllGather (2× shard for gather buffer)
	paramsPerStage := totalParams / float64(pp)
	paramsPerGPU := paramsPerStage / float64(fsdp)
	paramMem := paramsPerGPU * bpd
	// AllGather buffer: temporarily holds the full stage's params for one layer
	paramsPerLayer := totalParams / float64(model.NumLayers)
	allGatherBuffer := paramsPerLayer * bpd

	// Gradients: same shard, always fp32 for ReduceScatter accumulation
	gradMem := paramsPerGPU * 4.0

	// Optimizer: AdamW m + v, fp32 each, fully sharded
	optMem := paramsPerGPU * 8.0

	// Activations: per-layer, split by TP and PP
	layersPerStage := float64(model.NumLayers) / float64(pp)
	seqPerCP := float64(model.SeqLen) / float64(cp)
	actPerLayer := seqPerCP * float64(mb) * float64(model.HiddenDim) * 10.0 / float64(tp)
	actMem := actPerLayer * layersPerStage
	if actCkpt {
		// With activation checkpointing, only store activations at checkpoint
		// boundaries (every sqrt(L) layers) + recompute during backward
		actMem *= 1.0 / math.Sqrt(layersPerStage)
	}

	// MHA working memory: heads split by TP
	headsPerGPU := float64(model.NumHeads) / float64(tp)
	headDim := float64(model.HiddenDim) / float64(model.NumHeads)
	mhaMem := float64(mb) * headsPerGPU * seqPerCP * headDim * 2.0

	totalBytes := paramMem + allGatherBuffer + gradMem + optMem + actMem + mhaMem
	return totalBytes / 1e9
}

// ════════════════════════════════════════════════════════════════
//  Throughput estimation
// ════════════════════════════════════════════════════════════════

// estimateThroughput returns a relative score (higher = better).
// This is a heuristic that considers:
//  1. Compute: tokens per step (more = better)
//  2. TP overhead: AllReduce per layer on NVLink
//  3. PP overhead: pipeline bubble (PP-1)/PP
//  4. FSDP overhead: AllGather/ReduceScatter on IB (if spans nodes)
//  5. Activation checkpointing: ~33% more compute
//  6. FP8 vs BF16: ~2× compute throughput
func estimateThroughput(
	model aiv1.ModelArchSpec,
	gpu GPUSpec,
	tp, pp, fsdp, cp, mb int32,
	actCkpt bool,
	precision string,
	gpusPerNode int32,
) float64 {
	// Base: tokens processed per step
	seqPerCP := float64(model.SeqLen) / float64(cp)
	tokensPerStep := seqPerCP * float64(mb)

	// Compute multiplier: FP8 is ~1.5-2× faster than BF16 on Hopper
	computeMul := 1.0
	if precision == "fp8" {
		computeMul = 1.7
	}

	// Activation checkpointing penalty: ~33% more FLOPs (recompute forward in backward)
	if actCkpt {
		computeMul *= 0.75
	}

	// TP overhead: AllReduce on every layer's output
	// More TP = less memory but more communication
	// On NVLink this is fast, so penalty is modest
	tpEfficiency := 1.0
	if tp > 1 {
		tpOverheadFraction := 0.05 * float64(tp) / 8.0 // ~5% at TP=8 on NVLink
		tpEfficiency = 1.0 - tpOverheadFraction
	}

	// PP overhead: pipeline bubble = (PP-1) / (PP * num_microbatches)
	// With 1F1B schedule, bubble is (PP-1) microbatches out of total
	numMicrobatches := float64(mb) // simplified; in practice global_batch / micro_batch
	if numMicrobatches < float64(pp) {
		numMicrobatches = float64(pp) // need at least PP microbatches for pipeline
	}
	ppEfficiency := 1.0
	if pp > 1 {
		ppBubbleFraction := float64(pp-1) / (float64(pp) + numMicrobatches - 1)
		ppEfficiency = 1.0 - ppBubbleFraction
	}

	// FSDP overhead: AllGather + ReduceScatter per layer
	// If FSDP is intra-node only, it's on NVLink (fast)
	// If FSDP spans nodes, it's on IB (slower)
	fsdpEfficiency := 1.0
	if fsdp > 1 {
		fsdpSpansNodes := fsdp > (gpusPerNode / tp)
		if fsdpSpansNodes {
			paramsPerLayer := model.ParamsBillions * 1e9 / float64(model.NumLayers)
			fsdpVolPerLayer := 2.0 * paramsPerLayer * bytesPerDtype(precision) // AllGather + ReduceScatter
			fsdpTimePerLayer := fsdpVolPerLayer / (gpu.IBBWGBps * 1e9)
			// Estimate compute time per layer: ~1ms for a large model on H100
			computeTimePerLayer := 0.001
			fsdpOverlap := 0.7 // FSDP2 prefetches ~70% of communication
			fsdpEfficiency = 1.0 - (1.0-fsdpOverlap)*fsdpTimePerLayer/computeTimePerLayer
			if fsdpEfficiency < 0.5 {
				fsdpEfficiency = 0.5
			}
		} else {
			fsdpEfficiency = 0.98 // intra-node FSDP on NVLink is nearly free
		}
	}

	// Prefer larger micro-batch (better GPU utilization)
	batchEfficiency := 1.0 - 0.1/float64(mb) // mb=1: 0.9, mb=4: 0.975

	score := tokensPerStep * computeMul * tpEfficiency * ppEfficiency * fsdpEfficiency * batchEfficiency
	return score
}
