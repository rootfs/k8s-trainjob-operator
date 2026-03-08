---
name: model-eval
description: Evaluate trained model performance through benchmarking, detect regressions against baseline metrics, and validate checkpoint quality. Use when adding evaluation jobs, defining benchmark suites, tracking metrics across training runs, or building regression detection logic.
---

# Model Eval Agent

## Scope

You own the post-training evaluation and regression detection layer:

- `internal/controller/checkpoint.go` — checkpoint validation Job builder (you extend this with eval capabilities)
- `internal/webhook/model_registry.go` — baseline metrics per model (you add expected benchmark scores)
- `api/v1alpha1/types.go` — EvalSpec and eval-related status fields
- `examples/trainjob_sample.yaml` — eval-enabled example CRs

## What You Do

1. **Design the eval Job builder** — create a `buildEvalJob()` function in a new `internal/controller/eval.go` that runs post-training benchmarks against a checkpoint. The eval Job loads the checkpoint, runs a benchmark suite, and writes results to a structured output (JSON to stdout or a ConfigMap).

2. **Define benchmark suites** per model type:
   - **Embedding models** (the primary use case for vLLM Semantic Router): MTEB retrieval tasks (NDCG@10), STS correlation, classification accuracy
   - **Language models**: perplexity on a held-out set, few-shot accuracy on standard benchmarks (MMLU, HellaSwag, ARC)
   - **Multimodal models**: image-text retrieval (Recall@1/5/10), VQA accuracy

3. **Regression detection** — compare eval results against baselines stored in the model registry or in annotations on the TrainJob CR. Flag regressions when a metric drops below a configurable threshold (e.g., > 2% drop in NDCG@10 from baseline).

4. **Extend TrainJobSpec** with an optional `EvalSpec`:
   ```go
   type EvalSpec struct {
       Enabled         bool     `json:"enabled"`
       BenchmarkSuite  string   `json:"benchmarkSuite"`  // e.g., "mteb-retrieval", "mmlu", "custom"
       BaselineMetrics map[string]float64 `json:"baselineMetrics,omitempty"`
       RegressionThreshold float64 `json:"regressionThreshold,omitempty"` // default 0.02 (2%)
       EvalImage       string   `json:"evalImage,omitempty"` // defaults to training image
   }
   ```

5. **Add eval phase to the reconciler** — after training succeeds (PhaseSucceeded), optionally transition to a new `PhaseEvaluating` state that runs the eval Job, then to `PhaseEvalPassed` or `PhaseEvalFailed`.

## Eval Job Design

The eval Job should:

```python
# Pseudocode for the eval script
checkpoint = load_checkpoint(CHECKPOINT_PATH)
model = load_model(checkpoint, MODEL_NAME)

results = {}
for benchmark in BENCHMARK_SUITE:
    score = run_benchmark(model, benchmark)
    results[benchmark.name] = score

# Compare against baselines
regressions = []
for metric, score in results.items():
    if metric in BASELINES:
        delta = (BASELINES[metric] - score) / BASELINES[metric]
        if delta > REGRESSION_THRESHOLD:
            regressions.append({
                "metric": metric,
                "baseline": BASELINES[metric],
                "actual": score,
                "delta_pct": delta * 100
            })

if regressions:
    print(json.dumps({"status": "REGRESSION", "regressions": regressions}))
    sys.exit(1)
else:
    print(json.dumps({"status": "PASS", "results": results}))
    sys.exit(0)
```

The script receives configuration via environment variables:
- `CHECKPOINT_PATH` — path to the checkpoint directory
- `MODEL_NAME` — model identifier for registry lookup
- `BENCHMARK_SUITE` — comma-separated benchmark names or a suite preset
- `BASELINES` — JSON-encoded baseline metrics
- `REGRESSION_THRESHOLD` — float, default 0.02

## Regression Detection Patterns

### Absolute threshold
Flag if metric < baseline - threshold. Simple, but doesn't account for noise.

### Relative threshold (preferred)
Flag if (baseline - metric) / baseline > threshold. Handles different metric scales.

### Statistical
Track metric history across runs. Flag if the latest score is > 2 standard deviations below the rolling mean. Requires storing history — use annotations or a ConfigMap.

For the initial implementation, use **relative threshold** with a default of 2%. Statistical detection is a follow-up.

## Constraints

- The eval Job must use the same image as training (or a user-specified eval image) to ensure model loading compatibility.
- Eval Jobs should carry `kueue.x-k8s.io/queue-name: none` like other child resources.
- Eval should be optional — disabled by default, no impact on existing TrainJob CRs without EvalSpec.
- The eval Job needs GPU access (at least 1 GPU) to load and run the model.
- Eval results should be surfaced in TrainJobStatus (e.g., `status.evalResults` map and `status.evalPassed` bool).
- After changes, run `go build ./...` and `go vet ./...`.

## Files to Read First

1. `internal/controller/checkpoint.go` — the checkpoint validation Job builder (model for how to structure the eval Job)
2. `internal/controller/trainjob_controller.go` — reconciler state machine (where to add the eval phase)
3. `api/v1alpha1/types.go` — CRD types (where to add EvalSpec and eval status fields)
4. `internal/webhook/model_registry.go` — where baseline metrics per model would live
5. `examples/trainjob_sample.yaml` — add eval-enabled examples
