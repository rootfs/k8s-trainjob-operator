# CRD: TrainJob

**API Group:** `training.vsr.dev/v1alpha1`
**Kind:** `TrainJob`

## Spec

| Field | Type | Required | Description |
|---|---|---|---|
| `model` | string | yes | Model identifier (e.g., `llama-3-70b`). Used for registry lookups. |
| `image` | string | yes | Training container image. |
| `numNodes` | int32 | yes | Number of worker nodes. |
| `gpusPerNode` | int32 | yes | GPUs per node. Must be 1, 2, 4, or 8. |
| `tpDegree` | int32 | yes* | Tensor parallelism degree. Auto-set if `autoParallelism: true`. |
| `ppDegree` | int32 | yes* | Pipeline parallelism degree. Auto-set if `autoParallelism: true`. |
| `cpDegree` | int32 | no | Context parallelism (default 1). Useful for seq_len >= 32k. |
| `precision` | string | no | `bf16`, `fp8`, or `fp32`. Defaults to `bf16`. |
| `autoParallelism` | bool | no | Let the webhook derive TP/PP/CP/precision/batch automatically. |
| `modelSpec` | ModelArchSpec | no | Model architecture details for memory estimation. |
| `nodeSelector` | map | no | Standard node selector. GPU type is read from `nvidia.com/gpu.product`. |
| `checkpoint` | CheckpointSpec | yes | Checkpoint configuration. |
| `maxRuntime` | duration | no | Hard timeout for the job. |
| `skipProlog` | bool | no | Skip the GPU health check. Fine for short test jobs. |
| `enableSidecar` | *bool | no | Enable the GPU monitoring sidecar on worker pods. Defaults to true. |
| `suspend` | *bool | no | Hold job creation. Used by Kueue for suspend-based admission. Auto-set by webhook when `kueue.x-k8s.io/queue-name` label is present. |
| `command`, `args`, `env` | - | no | Container command overrides and extra environment variables. |
| `cpuPerNode`, `memPerNode` | quantity | no | CPU and memory requests per worker. |

FSDP degree is not a user-facing field — it's inferred as `totalGPUs / (TP × PP × CP)`.

## ModelArchSpec

Optional, but enables memory/bandwidth validation and is required for auto-parallelism on custom models not in the built-in registry.

| Field | Type | Description |
|---|---|---|
| `paramsBillions` | float64 | Total parameters in billions |
| `hiddenDim` | int64 | Hidden dimension |
| `numLayers` | int32 | Transformer layers |
| `numHeads` | int32 | Attention heads |
| `numKVHeads` | int32 | KV heads for GQA (0 = standard MHA) |
| `seqLen` | int64 | Max sequence length |
| `microBatchSize` | int32 | Per-GPU micro-batch size |
| `activationCheckpointing` | bool | Enable activation recomputation |

## CheckpointSpec

| Field | Type | Description |
|---|---|---|
| `enabled` | bool | Whether to checkpoint at all |
| `intervalMinutes` | *int32 | How often (defaults to 30) |
| `storagePath` | string | PVC path or S3 URI |
| `retainCount` | *int32 | Checkpoints to keep (defaults to 3) |
| `validateOnSave` | bool | Run a validation job after each save |

## Status

| Field | Description |
|---|---|
| `phase` | Current lifecycle phase (see [Workflow](design.md#workflow)) |
| `conditions` | Standard Kubernetes conditions |
| `startTime` / `completionTime` | Timestamps |
| `currentStep` | Training iteration (if reported by workers) |
| `lastCheckpoint` / `lastCheckpointTime` | Most recent valid checkpoint |
| `healthyNodes` / `totalNodes` | Nodes that passed prolog vs. allocated |
| `readyWorkers` | Worker pods in Ready state |
| `failureReason` / `failureMessage` | Why the job failed |

## Quick Example

Auto-parallelism for Llama 3 70B:

```yaml
apiVersion: training.vsr.dev/v1alpha1
kind: TrainJob
metadata:
  name: llama3-70b-auto
spec:
  model: llama-3-70b
  image: ghcr.io/vsr/llama:v2.4.0
  autoParallelism: true

  numNodes: 32
  gpusPerNode: 8

  nodeSelector:
    nvidia.com/gpu.product: "NVIDIA-H100-SXM5-80GB"

  checkpoint:
    enabled: true
    storagePath: /checkpoints
    validateOnSave: true

  maxRuntime: "168h"
  command: ["torchrun"]
  args:
    - "--nnodes=$(NNODES)"
    - "--nproc_per_node=$(NPROC_PER_NODE)"
    - "--rdzv_backend=c10d"
    - "--rdzv_endpoint=$(MASTER_ADDR):$(MASTER_PORT)"
    - "train.py"
```

The webhook fills in `tpDegree`, `ppDegree`, `cpDegree`, `precision`, and micro-batch based on the model's architecture and H100 hardware specs.

See [`examples/trainjob_sample.yaml`](../examples/trainjob_sample.yaml) for more examples including manual config, rejected configs, Kueue integration, and standalone mode.
