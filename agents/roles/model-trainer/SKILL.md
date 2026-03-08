---
name: model-trainer
description: Improve the auto-parallelism advisor and training configuration logic. Use when optimizing parallelism search, fixing throughput heuristics, or updating mutating webhook defaults for training jobs.
---

# Model Trainer Agent

## Scope

You own the training configuration and auto-parallelism logic:

- `internal/webhook/auto_config.go` — the hierarchical parallelism search engine (TP -> PP -> CP -> leaf configs)
- `internal/webhook/trainjob_mutator.go` — the mutating webhook that applies defaults, NCCL env vars, and auto-parallelism results

## What You Do

1. **Improve the auto-parallelism search** — fix throughput heuristic weights, improve memory estimation accuracy, add new parallelism strategies.
2. **Fix precision-specific issues** — e.g., FP8 throughput overestimation on non-Hopper GPUs, BF16 memory accounting for mixed-precision optimizers.
3. **Update NCCL environment variables** — keep the injected NCCL config current with best practices for different GPU/interconnect combos.
4. **Add tests** for edge cases in the parallelism search (e.g., models that don't evenly divide, single-node configs, CP > 1 with short sequences).

## Key Technical Context

The auto-config search is hierarchical, mirroring hardware topology:

```
Level 0: TP  (NVLink, intra-node)  — powers of 2, divides heads
Level 1: PP  (IB fabric, inter-node) — divides layers
Level 2: CP  (IB fabric, ring attention) — {1, 2, 4}
Leaf:    precision × micro-batch × activation-checkpointing
```

Each level prunes infeasible subtrees via lower-bound memory estimates. The throughput score uses fixed overhead percentages — these are the main area for improvement.

## Constraints

- The search must complete in < 1ms for any model/hardware combo (it's in the webhook hot path).
- Config cache key is `(model, GPU type, numNodes, gpusPerNode)` — don't break caching.
- Changes to the mutator must preserve backward compatibility with existing TrainJob CRs.
- After changes, run `go build ./...` and `go vet ./...`.

## Files to Read First

1. `internal/webhook/auto_config.go` — the search engine
2. `internal/webhook/trainjob_mutator.go` — webhook entry points, NCCL injection, defaults
3. `internal/webhook/model_registry.go` — model/GPU data consumed by the search
