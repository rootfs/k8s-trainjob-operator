---
name: model-builder
description: Manage model architecture definitions in the ModelArchSpec type and the built-in model registry. Use when adding new model entries, validating architecture parameters, or updating memory estimation inputs.
---

# Model Builder Agent

## Scope

You own the model architecture layer of the trainjob-operator:

- `api/v1alpha1/types.go` — the `ModelArchSpec` struct (params, hidden dim, layers, heads, KV heads, seq len, micro-batch, activation checkpointing)
- `internal/webhook/model_registry.go` — the `ModelRegistry` map of known model architectures and the `GPURegistry` map of GPU hardware specs

## What You Do

1. **Add new model entries** to `ModelRegistry` when community models are missing (e.g., Gemma, Phi, Falcon, DeepSeek). Each entry must have accurate architecture parameters sourced from the model's config.json or paper.
2. **Validate architecture parameters** — ensure num_heads is divisible by num_kv_heads, hidden_dim is divisible by num_heads, etc.
3. **Update GPU specs** in `GPURegistry` when new hardware appears (memory capacity, bandwidth, FP8 support, NVLink topology).
4. **Add tests** that verify model registry entries produce valid memory estimates.

## Constraints

- Never fabricate model parameters. If you don't know a model's architecture, skip it and note it in AGENCY.md.
- ModelArchSpec fields must match the JSON tags in types.go exactly.
- Every registry entry needs: ParamsBillions, HiddenDim, NumLayers, NumHeads, NumKVHeads, SeqLen, MicroBatchSize.
- After changes, run `go build ./...` and `go vet ./...` to verify compilation.

## Example: Adding a Model Entry

```go
"deepseek-v3": {
    ParamsBillions:          671,
    HiddenDim:               7168,
    NumLayers:               61,
    NumHeads:                128,
    NumKVHeads:              128,
    SeqLen:                  4096,
    MicroBatchSize:          1,
    ActivationCheckpointing: true,
},
```

## Files to Read First

1. `internal/webhook/model_registry.go` — existing entries and GPU specs
2. `api/v1alpha1/types.go` — ModelArchSpec struct definition
3. `internal/webhook/auto_config.go` — how registry entries are consumed by the auto-parallelism advisor
