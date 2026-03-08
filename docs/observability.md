# Observability

The operator exposes structured metrics across four lifecycle stages. These metrics are
machine-readable (stored in TrainJob status sub-resource) and agent-consumable — they
close the feedback loop that lets agents decide when to retrain, redeploy, or rollback.

## The Four Stages

```
┌────────────┐    ┌────────────┐    ┌────────────┐    ┌────────────┐
│  Training   │───►│    Eval     │───►│  Deploy     │───►│  Serving   │
│             │    │             │    │             │    │            │
│ status.     │    │ status.     │    │ status.     │    │ status.    │
│  training   │    │  eval       │    │  deployment │    │  serving   │
│             │    │             │    │             │    │            │
│ loss, MFU,  │    │ benchmarks, │    │ conversion, │    │ routing    │
│ gradients,  │    │ regression, │    │ load time,  │    │ accuracy,  │
│ throughput, │    │ quant       │    │ smoke test, │    │ drift,     │
│ hardware    │    │ sensitivity │    │ compat      │    │ latency,   │
│ events      │    │             │    │             │    │ A/B test   │
└────────────┘    └────────────┘    └────────────┘    └────────────┘
       │                 │                 │                 │
       └────────────────────────────────────────────────────┘
                    Feedback: each stage's signals drive
                    decisions in earlier stages
```

## Stage 1: Training Metrics

**Source**: metrics-collector sidecar reads a JSON file written by the training script.

**Enable**: Set `spec.metricsConfig.enabled: true` in the TrainJob CR.

**How it works**:
1. The training script writes metrics to `$METRICS_OUTPUT_FILE` (default `/var/run/training/metrics.json`)
2. The metrics-collector sidecar reads this file every `scrapeIntervalSeconds` (default 30)
3. On rank-0 pods, the sidecar patches the TrainJob status via the Kubernetes API
4. All pods export Prometheus metrics on a configurable port (default 9402)

**Metrics file schema** (what the training script writes):

```json
{
  "step": 12400,
  "total_steps": 50000,
  "train_loss": 0.342,
  "val_loss": 0.389,
  "gradient_norm": 1.24,
  "gradient_norm_max": 3.8,
  "zero_gradient_pct": 0.02,
  "tokens_per_second": 48200,
  "samples_per_second": 12.4,
  "mfu": 0.41,
  "comm_compute_ratio": 0.28,
  "step_time_sec": 1.82,
  "peak_memory_gb": 74.2,
  "allocated_memory_gb": 68.1,
  "oom_events": 0,
  "ecc_errors": 0,
  "thermal_events": 0,
  "nvlink_errors": 0
}
```

**Status fields** (`status.training`):

| Field | Type | What it tells you |
|---|---|---|
| `trainLoss` / `valLoss` | float64 | Convergence signal. Agents watch for plateau or divergence. |
| `lossDeltaPerK` | float64 | Loss improvement per 1000 steps. Flat = time to stop. |
| `gradientNorm` / `gradientNormMax` | float64 | Gradient health. Exploding norms = bad LR or parallelism. |
| `zeroGradientPct` | float64 | Fraction of zero gradients. High = vanishing gradients. |
| `tokensPerSecond` / `samplesPerSecond` | float64 | Training throughput. |
| `mfu` | float64 | Model FLOPS utilization. Measures how well the parallelism config uses the hardware. |
| `commComputeRatio` | float64 | Time in NCCL collectives / total step time. High ratio = too much inter-node traffic. |
| `stepTimeSec` | float64 | Wall-clock seconds per step. |
| `peakMemoryGB` / `allocatedGB` | float64 | GPU memory usage. |
| `oomEvents` | int32 | OOM event count. |
| `stepTimeP50Sec` / `stepTimeP99Sec` | float64 | Per-worker step time distribution (for straggler detection). |
| `stragglerRatio` | float64 | P99/P50 step time ratio. >2x = stragglers present. |
| `eccErrors` / `thermalEvents` / `nvlinkErrors` | int32 | Hardware event counters. |

**Prometheus metrics** (exported by the sidecar on the training metrics port):

All metrics use the `trainjob_` prefix and carry `trainjob`, `pod`, and `node` labels.

## Stage 2: Eval Metrics

**Source**: eval Job writes results to `/var/run/eval/results.json`.

**Enable**: Set `spec.evalConfig` in the TrainJob CR.

**How it works**:
1. After all training workers complete, the reconciler transitions to `PhaseEvaluating`
2. An eval Job is created using the training image (or a custom eval image)
3. The eval Job loads the checkpoint, runs benchmarks, and writes structured results
4. Eval failure doesn't fail the TrainJob — training still succeeded. The failure is recorded in a condition.

**Status fields** (`status.eval`):

| Field | Type | What it tells you |
|---|---|---|
| `benchmarks` | []BenchmarkResult | Per-benchmark scores with regression comparison. |
| `benchmarks[].previousValue` / `delta` | float64 | Change from previous model. Negative delta = regression. |
| `benchmarks[].passed` | bool | Whether this benchmark meets its threshold. |
| `quantizationSensitivity` | map[string]float64 | How metrics degrade under fp16/int8/fp8. |
| `inferenceLatencyP50Ms` / `P99Ms` | float64 | Model speed at eval time (single-request). |
| `embeddingQuality` | EmbeddingQualityMetrics | Embedding space health for retrieval models. |
| `verdict` | string | Agent recommendation: "promote", "rollback", "retrain". |

## Stage 3: Deployment Metrics

**Source**: model-install pipeline writes metrics after checkpoint conversion.

**Status fields** (`status.deployment`):

| Field | Type | What it tells you |
|---|---|---|
| `conversionTimeSec` | float64 | How long checkpoint→safetensors conversion took. |
| `outputSizeGB` | float64 | Converted model size. |
| `maxAbsError` | float64 | Max precision loss vs original weights. High = quantization too aggressive. |
| `loadTimeSec` | float64 | Time to load model into vLLM. |
| `servingMemoryGB` | float64 | GPU memory consumed after loading. |
| `timeToFirstTokenMs` | float64 | Time to first inference response. |
| `smokeTestPassed` | bool | Did the model produce correct outputs on known inputs? |
| `vllmVersion` | string | vLLM version tested against. |

## Stage 4: Serving Metrics

**Source**: scraped from vLLM Prometheus metrics or a serving observer.

**Status fields** (`status.serving`):

| Field | Type | What it tells you |
|---|---|---|
| `routingAccuracy` | float64 | Current routing accuracy in semantic router. |
| `routingAccuracyP7d` | float64 | 7-day rolling routing accuracy. |
| `embeddingDriftKL` / `embeddingDriftMMD` | float64 | Distribution shift from training data. Rising = model seeing OOD inputs. |
| `latencyP50Ms` / `P95Ms` / `P99Ms` | float64 | Inference latency under production load. |
| `requestsPerSecond` | float64 | Serving throughput. |
| `cacheHitRate` | float64 | Embedding cache effectiveness. |
| `abTestActive` / `abCurrentValue` / `abBaselineValue` / `abPValue` | - | A/B test comparison when running two model versions side-by-side. |
| `reformulationRate` | float64 | User query reformulation rate (proxy for result quality). |

## Feedback Loops

The four stages form backward arrows where later signals drive earlier decisions:

| Signal | Consumer | Action |
|---|---|---|
| `serving.routingAccuracy` drops 5% | model-eval agent | Flag regression, recommend retraining |
| `serving.embeddingDriftKL > threshold` | model-trainer agent | Schedule retraining on recent data |
| `eval.verdict == "rollback"` | model-install agent | Skip deployment, notify ops |
| `eval.quantizationSensitivity.fp8` too low | model-install agent | Use fp16 instead of fp8 for conversion |
| `training.mfu < 0.35` | infra agent | Adjust parallelism config in auto-advisor |
| `training.commComputeRatio > 0.4` | infra agent | Reduce TP, increase PP to reduce comm |
| `training.stragglerRatio > 2.0` | sre agent | Investigate slow nodes, update prolog checks |
| `deployment.smokeTestPassed == false` | model-install agent | Abort deployment, investigate compatibility |
| `deployment.maxAbsError > threshold` | model-install agent | Reduce quantization aggressiveness |

## Example: Metrics-Enabled TrainJob

```yaml
apiVersion: training.vsr.dev/v1alpha1
kind: TrainJob
metadata:
  name: clip-vit-l-v2
spec:
  model: clip-vit-l
  image: ghcr.io/vsr/clip-training:v1.2
  numNodes: 8
  gpusPerNode: 8
  autoParallelism: true

  nodeSelector:
    nvidia.com/gpu.product: "NVIDIA-H100-SXM5-80GB"

  checkpoint:
    enabled: true
    storagePath: /checkpoints
    validateOnSave: true

  metricsConfig:
    enabled: true
    scrapeIntervalSeconds: 15

  evalConfig:
    datasetPath: /data/eval/mteb-retrieval
    previousModelPath: /models/clip-vit-l-v1/model.safetensors
    gpusPerNode: 1

  maxRuntime: "48h"
```

This TrainJob will:
1. Auto-configure parallelism at admission
2. Run GPU health checks (prolog)
3. Train with live metrics collection (step time, loss, MFU, gradients)
4. After training completes, run evaluation against the MTEB retrieval benchmark
5. Compare eval results against the previous v1 model
6. Write a verdict ("promote"/"rollback"/"retrain") to `status.eval.verdict`
