# TrainJob Operator

A Kubernetes operator for managing distributed GPU training jobs. It handles the full lifecycle: admission-time validation, pre-training GPU health checks, worker orchestration via StatefulSets, checkpoint management, and per-pod GPU monitoring sidecars.

The original motivation was training multimodal embedding models for [vLLM Semantic Router](https://github.com/vllm-project/semantic-router) — things like fine-tuning CLIP, SigLIP, or BGE-style encoders that power the routing layer. Managing distributed GPU training for these models on Kubernetes had enough rough edges that it made sense to build an operator around it. The scope grew from there into a more general-purpose training job operator.

Built with [controller-runtime](https://github.com/kubernetes-sigs/controller-runtime) and [Kubebuilder](https://book.kubebuilder.io/) conventions.

> **Disclaimer**: This is a hobby / exploration project. It is **not** production-ready and is not affiliated with any company. The code was written to learn and experiment with Kubernetes operator patterns for GPU training workloads. There are known gaps (see [Limitations](#limitations) and [What's Missing for Production](#whats-missing-for-production)). Use at your own risk.

---

## Table of Contents

- [Problem Statement](#problem-statement)
- [Overview](#overview)
- [CRD: TrainJob](#crd-trainjob)
- [Architecture](#architecture)
- [Workflow](#workflow)
- [Auto-Parallelism Advisor](#auto-parallelism-advisor)
- [Prolog (Pre-Training Health Check)](#prolog-pre-training-health-check)
- [GPU Monitoring Sidecar](#gpu-monitoring-sidecar)
- [Examples](#examples)
- [Kueue Integration](#kueue-integration)
- [Comparison with Existing Frameworks](#comparison-with-existing-frameworks)
- [Limitations](#limitations)
- [What's Missing for Production](#whats-missing-for-production)
- [Project Structure](#project-structure)

---

## Problem Statement

Kubernetes gives you primitives — Pods, Jobs, StatefulSets — but distributed GPU training needs a lot of domain-specific glue between "I want to train this model" and "training completed successfully." This operator tries to own that glue.

The pain points it targets:

### Configuration correctness (usability)

Getting TP/PP/FSDP/CP right for a given model and hardware combo is genuinely hard. People misconfigure it, wait 20-30 minutes for scheduling and startup, then hit OOM or terrible throughput. The validating webhook and auto-parallelism advisor are about catching that at admission time — fail fast, not fail late.

### Hardware reliability

At scale, bad GPUs, degraded NVLink, flaky InfiniBand ports are not rare — they're expected. The prolog is about not burning hundreds of GPU-hours only to discover node 17 has a bad GPU 45 minutes into training. It's a pre-flight check.

### Runtime observability

Once training is running, you need to know about stragglers, hardware anomalies, and step-time regressions before they snowball into a job failure. The monitoring sidecar is about that.

### Checkpoint lifecycle

Checkpoints are your insurance policy against wasted compute. The operator manages the save/validate/retain cycle and protects the last checkpoint on deletion — so a failed job doesn't also lose your progress.

### Model-infra co-design

The auto-parallelism advisor is explicitly about matching model architecture to hardware. It takes model parameters (hidden dim, layers, heads, sequence length) and GPU specs (memory, bandwidth, FP8 support) and finds a good parallelism config. That's model-infra co-design in a narrow, practical sense.

### What this is *not*

This is not a scheduler (that's Kueue/Volcano), not a training framework (that's PyTorch/NeMo), and not a cluster provisioner. It's the **operational layer in between** — an opinionated lifecycle operator that tries to make GPU training jobs less painful to run on Kubernetes.

---

## Overview

The operator introduces a single CRD, `TrainJob`, that captures everything needed to run a multi-node GPU training job: model name, cluster shape, parallelism strategy, checkpoint policy, and more. On top of that, it layers:

- **Mutating webhook** — injects NCCL environment variables, sets defaults, and optionally auto-configures parallelism (TP/PP/FSDP/CP) based on model architecture and GPU hardware.
- **Validating webhook** — catches bad configs at admission time (e.g., TP > GPUs per node, FP8 on non-Hopper GPUs, memory estimation failures) so you don't waste 30 minutes waiting for OOM.
- **State-machine reconciler** — drives the job through phases: suspended → prolog health check → worker StatefulSet creation → monitoring → checkpoint management → completion. Supports Kueue's suspend-based admission control.
- **GPU monitoring sidecar** — a gpu-monitor container is included directly in the worker StatefulSet pod template (no cluster-wide pod webhook needed). Provides DCGM metrics, straggler detection, and anomaly watching.

---

## CRD: TrainJob

**API Group:** `training.vsr.dev/v1alpha1`
**Kind:** `TrainJob`

### Spec

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

### ModelArchSpec

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

### CheckpointSpec

| Field | Type | Description |
|---|---|---|
| `enabled` | bool | Whether to checkpoint at all |
| `intervalMinutes` | *int32 | How often (defaults to 30) |
| `storagePath` | string | PVC path or S3 URI |
| `retainCount` | *int32 | Checkpoints to keep (defaults to 3) |
| `validateOnSave` | bool | Run a validation job after each save |

### Status

| Field | Description |
|---|---|
| `phase` | Current lifecycle phase (see [Workflow](#workflow)) |
| `conditions` | Standard Kubernetes conditions |
| `startTime` / `completionTime` | Timestamps |
| `currentStep` | Training iteration (if reported by workers) |
| `lastCheckpoint` / `lastCheckpointTime` | Most recent valid checkpoint |
| `healthyNodes` / `totalNodes` | Nodes that passed prolog vs. allocated |
| `readyWorkers` | Worker pods in Ready state |
| `failureReason` / `failureMessage` | Why the job failed |

---

## Architecture

```
                     User creates TrainJob CR
                               │
                 ┌─────────────┴─────────────┐
                 ▼                             ▼
        Mutating Webhook              Validating Webhook
        ├─ Auto-parallelism           ├─ TP ≤ GPUs/node?
        │  (if enabled)               ├─ FP8 on Hopper+ only?
        ├─ NCCL env vars              ├─ Checkpoint for long jobs?
        ├─ Default precision=bf16     ├─ GPU memory estimation
        └─ Default checkpoint config  └─ Head divisibility by TP
                 │                             │
                 └─────────────┬───────────────┘
                               │ (admitted)
                               ▼
                       API Server persists CR
                               │
                               ▼
                 ┌─────────────────────────────┐
                 │   Reconciler State Machine   │
                 │                              │
                 │   Suspended (if suspend=true)│
                 │     └─► (Kueue unsuspends)   │
                 │           ▼                  │
                 │   Pending                    │
                 │     └─► PrologRunning        │
                 │           ├─► PrologPassed   │
                 │           │     └─► Running  │
                 │           │          ├─► Succeeded
                 │           │          ├─► Checkpointing ─► Running
                 │           │          └─► Failed
                 │           └─► PrologFailed ─► Failed
                 └─────────────────────────────┘
                               │
                    StatefulSet creates Pods
                    (with gpu-monitor sidecar
                     built into pod template)
```

---

## Workflow

Here's what happens end-to-end when you `kubectl apply` a TrainJob:

### 1. Admission (webhooks)

The **mutating webhook** fires first:
- If `autoParallelism: true`, it resolves the model architecture (from `modelSpec` or the built-in model registry) and the GPU spec (from `nodeSelector`), then brute-force searches ~1,600 parallelism configurations to find the highest-throughput combo that fits in GPU memory. The chosen config is written back into the spec and annotated on the CR.
- Injects NCCL environment variables tuned for the GPU type (H100 gets IB/HCA settings, A100 gets a different set, L40/L4 gets socket-based config).
- Sets defaults: `precision: bf16`, `cpDegree: 1`, `checkpoint.intervalMinutes: 30`, `checkpoint.retainCount: 3`.

Then the **validating webhook** checks ~10 rules. If any fail, the create request is rejected immediately with a specific error message. No waiting for scheduling, no OOM after 20 minutes.

### 2. Prolog (GPU health check)

Unless `skipProlog: true`, the reconciler creates an indexed Kubernetes Job — one pod per node, all running in parallel — using the same training image. Each pod runs three phases:

- **Hardware health**: Checks GPU count, ECC errors, runs DCGM diagnostics, verifies clock frequencies, checks PCIe link width/speed, and InfiniBand port status (on IB-capable GPUs).
- **Kernel validation**: Runs `compute-sanitizer` memcheck on a CUDA matmul, warms up `torch.compile` (verifies Triton codegen on the actual hardware), runs a precision smoke test (bf16/fp8 matmul vs fp32 reference), and checks cross-GPU consistency (same matmul should produce the same result on every GPU — catches silent data corruption).
- **Interconnect bandwidth**: Measures P2P bandwidth between all GPU pairs on the node to detect degraded NVLink or asymmetric link failures.

The reconciler polls the prolog Job every 15 seconds. If it fails, the TrainJob transitions to `Failed` with the specific failure reason from the prolog output.

### 3. Worker creation

On prolog success (or if skipped), the reconciler creates:
- A **headless Service** on port 29500 for `torch.distributed` DNS-based rendezvous.
- A **StatefulSet** with `numNodes` replicas and `ParallelPodManagement` (all pods start simultaneously, no rolling). Each pod gets environment variables for distributed training: `MASTER_ADDR`, `WORLD_SIZE`, `NNODES`, `NPROC_PER_NODE`, `TP_DEGREE`, `PP_DEGREE`, `FSDP_DEGREE`, `CP_DEGREE`, `PRECISION`, `CHECKPOINT_DIR`, and `POD_INDEX` derived from the StatefulSet ordinal.

StatefulSets (rather than Deployments) are used because distributed training needs stable pod identities for rank assignment.

### 4. Monitoring

While running, the reconciler polls the StatefulSet every 60 seconds, updating `status.readyWorkers`. If a pod fails and checkpointing is enabled with `currentStep > 0`, the job transitions to `Checkpointing` (which can optionally run a validation job on the checkpoint). If max runtime is exceeded, the job transitions to `Failed`.

### 5. GPU monitoring sidecar

When `enableSidecar` is true (the default), the reconciler includes a `gpu-monitor` sidecar container directly in the worker StatefulSet's pod template. This replaces the earlier approach of injecting the sidecar via a cluster-wide pod mutating webhook — that design required a `MutatingWebhookConfiguration` on all pods in the cluster, which creates the scaling and blast-radius problems described in [Webhook Efficiency](#webhook-efficiency).

By building the sidecar into the StatefulSet template, the operator is fully self-service: it only needs CRD-scoped webhooks (for TrainJob resources), with no cluster-wide pod interception.

The sidecar provides:

| Subsystem | Purpose |
|---|---|
| DCGM Exporter | Prometheus GPU metrics on :9400 |
| Health Endpoint | Liveness check on :9401 |

Prometheus scrape annotations (`prometheus.io/scrape`, `prometheus.io/port`, `prometheus.io/path`) are set on the pod template when the sidecar is enabled. The sidecar can be disabled per-job with `enableSidecar: false` for lightweight dev runs.

> **Note**: The `PodSidecarInjector` webhook code is retained in `internal/webhook/` as a legacy option for clusters that prefer webhook-based injection. It is no longer registered by default in `main.go`.

### 6. Deletion

A finalizer (`training.vsr.dev/checkpoint-protection`) ensures the last checkpoint path is preserved before the TrainJob and its child resources are garbage collected.

---

## Auto-Parallelism Advisor

This is the most interesting part of the operator. Instead of manually figuring out TP/PP/FSDP/CP for a given model and cluster, you set `autoParallelism: true` and the mutating webhook figures it out.

**How it works:**

1. Resolve the model architecture — either from `modelSpec` in the CR, or by looking up `spec.model` in a built-in model registry (Llama 3 family, Mixtral, Mistral, etc.).
2. Resolve the GPU hardware from `nodeSelector["nvidia.com/gpu.product"]` → a GPU registry with memory capacity, bandwidth, FP8 support, etc.
3. Enumerate all valid combinations:
   - TP: powers of 2 that divide both `numHeads` and `numKVHeads`, up to `gpusPerNode`
   - PP: divisors of `numLayers`, up to `numNodes`
   - CP: {1, 2, 4} (only useful for long sequences)
   - Micro-batch: {1, 2, 4, 8}
   - Activation checkpointing: {true, false}
   - Precision: {bf16, fp8} (fp8 only on Hopper+)
4. For each combo, estimate per-GPU memory usage. Discard anything that exceeds GPU capacity.
5. Score the remaining configs by a throughput heuristic that accounts for: TP AllReduce overhead, PP pipeline bubble (1F1B schedule), FSDP AllGather/ReduceScatter cost, activation recompute overhead, FP8 speedup, and micro-batch GPU utilization.
6. Return the highest-scoring config.

The search space is small (~1,920 upper bound) so brute-force is fine — it runs in microseconds.

**Important caveats**: The throughput model is a heuristic. It uses fixed overhead percentages (e.g., ~5% TP overhead at TP=8, ~70% FSDP overlap, ~1.7x FP8 speedup) that are rough approximations. Real performance depends on network topology, driver versions, framework implementation details, and many other factors. This is a starting point, not a replacement for profiling.

### Hierarchical search and pruning

The search is structured to mirror the hardware hierarchy, not as a flat loop:

```
Level 0: TP  (NVLink tier, intra-node)
  └─ Level 1: PP  (IB fabric tier, inter-node pipeline)
       └─ Level 2: CP  (IB fabric tier, ring attention)
            └─ Leaf: precision × micro-batch × act-ckpt
```

At each level, a lower-bound memory estimate prunes infeasible subtrees before descending. For example, if a 70B model with TP=2 can't fit even with maximum PP, activation checkpointing enabled, and micro-batch=1, then the entire TP=2 branch is skipped — no need to evaluate any of the ~200 leaf configs underneath it.

This matters because the parallelism dimensions are not independent — they map directly to the physical interconnect hierarchy:

| Dimension | Interconnect | Constraint | Decides |
|---|---|---|---|
| TP | NVLink (900 GB/s) | ≤ GPUs per node, divides heads | How many GPUs share each layer's computation |
| PP | IB fabric (400 Gbps) | Divides num_layers, ≤ num_nodes | How model layers are split across nodes |
| CP | IB fabric | Only useful for seq_len ≥ 32k | How sequence is split for ring attention |
| FSDP | IB fabric | Derived: total / (TP × PP × CP) | How optimizer state and gradients are sharded |

Deciding TP first (the most constrained dimension) and pruning early avoids wasting time on configs that can't possibly fit. For a 405B model on H100s, this prunes ~60% of the search space at Level 0.

### Config caching

The mutator caches search results keyed by `(model, GPU type, numNodes, gpusPerNode)`. When multiple TrainJobs target the same model on the same hardware (common in hyperparameter sweeps or repeated runs), the second submission hits the cache and skips the search entirely.

---

## Webhook Efficiency

Admission webhooks are a well-known scaling bottleneck in Kubernetes. Mutating webhooks run serially in the kube-apiserver admission chain (not in parallel like validating webhooks), each with a default 10-second timeout. When a webhook intercepts resources it doesn't need to process, it wastes kube-apiserver QPS budget on no-ops.

This is not a theoretical problem. Existing projects have hit it at scale:

- **Kueue** ([#4141](https://github.com/kubernetes-sigs/kueue/issues/4141), [#5260](https://github.com/kubernetes-sigs/kueue/issues/5260)): Kueue's mutating webhook intercepts *all* Jobs across *all* namespaces when `manageJobsWithoutQueueName` is enabled. Unrelated CronJobs, Helm hooks, and CI/CD jobs all route through Kueue's webhook. Users reported CronJobs failing with "context deadline exceeded" because the webhook was overwhelmed or restarting — since `failurePolicy: Fail`, the entire cluster's Job creation blocks. Kueue is moving toward namespace-level opt-in and eventually MutatingAdmissionPolicy (runs inside kube-apiserver, no network hop).

- **Volcano** ([#4127](https://github.com/volcano-sh/volcano/issues/4127), [#3358](https://github.com/volcano-sh/volcano/issues/3358)): Volcano registered webhooks for Pods, PodGroups, and Jobs — all enabled by default. At scale, users hit `context deadline exceeded`. Volcano's response: [disable most webhooks by default](https://github.com/volcano-sh/volcano/issues/4127) and replace patch-based defaulting with kubebuilder semantic defaults to eliminate the webhook entirely. Their stated principle: *"webhooks can greatly affect performance in large scale scenario, so we should avoid use webhooks as little as possible."*

- **YuniKorn** ([YUNIKORN-3075](https://issues.apache.org/jira/browse/YUNIKORN-3075)): YuniKorn's admission controller mutates every pod to add `schedulerName: yunikorn`. After redeployment, a bug caused webhooks to be recreated with cluster-level scope instead of namespaced, mutating all pods unintentionally.

### What this operator does differently

Given these lessons, the TrainJob operator is designed to minimize webhook blast radius:

**1. TrainJob webhooks only fire on TrainJob resources — not on Jobs or Pods.**

The CRD-level webhooks (mutating + validating) are scoped to `groups=training.vsr.dev, resources=trainjobs`. They never intercept batch Jobs, CronJobs, Deployments, or any other resource. Even in a cluster running thousands of workloads, these webhooks only fire on TrainJob create/update — which is low volume (one per job submission, not one per pod).

This is the key difference from Kueue and Volcano's approach, where the webhooks catch broad resource types (`batch/v1 Jobs`, `v1 Pods`) and then filter. Here, the kube-apiserver itself does the filtering before the webhook is called, because the webhook registration specifies the exact API group and resource.

**2. No cluster-wide pod webhook.**

The GPU monitoring sidecar is now included directly in the worker StatefulSet's pod template by the reconciler. This eliminates the need for a `MutatingWebhookConfiguration` on pods entirely — no cluster-wide webhook intercepting every pod creation.

The earlier design used a pod webhook with `failurePolicy: Ignore` and O(1) label-based early exit, which worked but still added a network hop for every pod create in the cluster. The current approach has zero webhook overhead for non-TrainJob pods.

The legacy `PodSidecarInjector` is retained in the codebase for clusters that prefer webhook-based injection, but it is not registered by default.

**3. Auto-parallelism results are cached.**

The brute-force search (~microseconds) is cached by `(model, GPU type, numNodes, gpusPerNode)`. Hyperparameter sweeps that submit 50 TrainJobs with the same model and cluster shape pay the search cost once.

**4. No webhook for the prolog or worker resources.**

The prolog Job, headless Service, and worker StatefulSet are created by the reconciler (server-side), not by the user. They never go through admission webhooks. This is a deliberate design choice: the reconciler owns these child resources and sets them up correctly, so there's nothing to validate or mutate at admission time.

### What we'd still want to improve

- **MutatingAdmissionPolicy for TrainJob defaults**: Once MAPs reach beta, the TrainJob defaulting (NCCL env vars, checkpoint defaults) could run as an in-process CEL policy inside kube-apiserver — no external webhook server needed for the simple cases. The auto-parallelism search would still need a webhook since it's too complex for CEL.

---

## Kueue Integration

The operator supports coexistence with [Kueue](https://github.com/kubernetes-sigs/kueue) via a **suspend-based admission pattern**. This is the same pattern Kueue uses for batch Jobs: the workload starts suspended, Kueue holds it until quota is available, then unsuspends it to start.

### How it works

1. **User submits a TrainJob** with a `kueue.x-k8s.io/queue-name` label pointing to a Kueue LocalQueue.
2. **Mutating webhook** detects the label and auto-sets `spec.suspend: true` (if not already set by the user).
3. **Reconciler** sees `suspend=true` and enters the `Suspended` phase — no prolog Job, no StatefulSet, no headless Service. The TrainJob CR exists but produces no child resources.
4. **Kueue** sees the TrainJob workload (via a Kueue integration or generic job webhook). When quota is granted, Kueue sets `spec.suspend: false`.
5. **Reconciler** detects the unsuspend, transitions to `Pending`, and proceeds normally: prolog → workers → running.

### Child resource isolation

The prolog Job, checkpoint validation Job, and worker StatefulSet are created by the reconciler — they're not user-submitted. Kueue's webhook intercepts Jobs by default (and in some configurations, all pods), which would cause it to try to manage these internal resources.

To prevent this, all child Jobs carry a `kueue.x-k8s.io/queue-name: none` label. This signals to Kueue that these are operator-managed internal resources and should not be admitted through Kueue's queue. The TrainJob itself is the unit of admission — its children inherit its quota grant implicitly via the suspend/unsuspend handshake.

### Without Kueue

If no `kueue.x-k8s.io/queue-name` label is present, `spec.suspend` stays nil (falsy) and the reconciler starts immediately. This is the standalone mode for dedicated clusters without a queueing system.

### What this doesn't solve

- **Gang scheduling**: Kueue provides admission control (when to start), not gang scheduling (ensure all pods are co-scheduled). The operator still relies on StatefulSet pod management, which is best-effort. For true gang scheduling, you'd need Kueue + a scheduler plugin or Volcano.
- **TAS (Topology-Aware Scheduling)**: Kueue's TAS can place the TrainJob's pods with rack/switch awareness, but the operator doesn't set the topology annotations. This is a gap — you'd configure TAS at the Kueue ClusterQueue level.
- **Webhook stacking**: If Kueue, the TrainJob operator, and other controllers all register mutating webhooks for overlapping resource types, the kube-apiserver runs them serially. The operator minimizes this by only registering webhooks for TrainJob resources (not Jobs or Pods), but the interaction between Kueue's Job webhook and the operator's child Jobs is why the `kueue.x-k8s.io/queue-name: none` label exists.

### Example

See `examples/trainjob_sample.yaml` (Example 8) for a complete Kueue-integrated TrainJob.

---

## Prolog (Pre-Training Health Check)

The prolog is a Kubernetes indexed Job that runs one pod per node before training starts. The idea is to catch bad GPUs, degraded links, or driver issues before you commit hundreds of GPUs to a multi-hour training run.

It runs in three phases per node:
1. **Hardware**: nvidia-smi checks, ECC errors, DCGM diagnostics, PCIe/IB verification
2. **Kernel**: CUDA memcheck, torch.compile warmup, precision smoke tests, cross-GPU consistency
3. **Interconnect**: P2P bandwidth matrix between all GPU pairs

The prolog uses the actual training image, not a generic test image. This catches software stack issues (wrong CUDA version, missing libraries, etc.) in addition to hardware problems.

---

## Examples

See [`examples/trainjob_sample.yaml`](examples/trainjob_sample.yaml) for:

- **Manual config**: 70B model on 32 H100 nodes with explicit TP=8, PP=4
- **Small fine-tune**: 8B model on 2 A100 nodes with `skipProlog: true`
- **Auto-parallelism**: 70B, custom 13B, and 405B models where the webhook derives the parallelism config
- **Rejected configs**: Examples that the validating webhook catches (TP > GPUs/node, FP8 on A100, OOM, head count mismatch)
- **Kueue integration**: TrainJob with `kueue.x-k8s.io/queue-name` label for suspend-based admission
- **Standalone mode**: TrainJob without Kueue, with sidecar disabled

Quick example — auto-parallelism for Llama 3 70B:

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

---

## Comparison with Existing Frameworks

This section tries to be fair about what this project does and doesn't do relative to established tools.

### vs. Kubeflow Training Operator

[Kubeflow Training Operator](https://github.com/kubeflow/training-operator) is the standard way to run distributed training on Kubernetes. It provides CRDs like `PyTorchJob`, `TFJob`, `MPIJob`, etc.

| Aspect | Kubeflow Training Operator | TrainJob Operator |
|---|---|---|
| Maturity | Production-grade, widely adopted, CNCF project | Hobby project, not battle-tested |
| Framework support | PyTorch, TensorFlow, MPI, XGBoost, PaddlePaddle, JAX | Framework-agnostic (anything torchrun-compatible) |
| Elastic training | Supports TorchElastic via `ElasticPolicy` | Not supported |
| GPU health checks | Not built-in | Built-in prolog Job (hardware, kernel, interconnect) |
| Auto-parallelism | Not built-in | Built-in heuristic advisor (with caveats — see above) |
| Checkpoint management | Not built-in (left to the training script) | Operator-managed: periodic save, validation, retention |
| Admission validation | Basic (schema validation) | Domain-specific rules (TP/PP constraints, memory estimation, FP8 GPU checks) |
| GPU monitoring sidecar | Not built-in | Injected via pod webhook (DCGM, straggler detection, anomaly watching) |
| Gang scheduling | Integrates with Volcano / Kueue | Not supported (relies on StatefulSet parallel pod management) |
| Community & support | Large community, good docs | Just me |

**Bottom line**: Kubeflow Training Operator is the right choice for production. This project explores ideas that Kubeflow doesn't cover out of the box (prolog checks, auto-parallelism, monitoring sidecar), but it lacks the breadth, maturity, and community that Kubeflow has.

### vs. Kueue

[Kueue](https://github.com/kubernetes-sigs/kueue) is a job queueing system for Kubernetes. It handles admission control, resource quotas, fair sharing, priority, and preemption. It also supports Topology Aware Scheduling (TAS) for placing workloads on nodes with good network locality (e.g., same rack, same IB switch).

These two projects solve different problems:

| Aspect | Kueue | TrainJob Operator |
|---|---|---|
| Focus | "When and where should this job run?" | "How should this training job be configured and monitored?" |
| Queueing / fair sharing | Yes — ClusterQueues, LocalQueues, ResourceFlavors | No queueing at all |
| Quota management | Yes — limits per namespace, team, cohort | None |
| Preemption | Yes — priority-based preemption | None |
| Topology-aware scheduling | Yes (TAS) — rack/block awareness for IB locality | None |
| Multi-cluster | Yes (MultiKueue) | No |
| Training-specific logic | None — it's framework-agnostic | Parallelism validation, auto-config, GPU health checks, checkpoints |

**Bottom line**: Kueue and a training operator are complementary, not competing. In a real setup you'd use Kueue for admission control and scheduling, and something like this operator (or Kubeflow) for the training-specific lifecycle. This project supports suspend-based Kueue integration — see [Kueue Integration](#kueue-integration).

### vs. Volcano

[Volcano](https://github.com/volcano-sh/volcano) is a batch scheduling system for Kubernetes. Its main feature for training workloads is **gang scheduling** — ensuring all pods in a distributed job are scheduled simultaneously (or not at all), avoiding deadlocks where half the pods are scheduled and the other half are waiting.

| Aspect | Volcano | TrainJob Operator |
|---|---|---|
| Focus | Batch scheduling, gang scheduling, fair sharing | Training job lifecycle management |
| Gang scheduling | Yes — core feature (`minAvailable` in PodGroup) | No — relies on StatefulSet pod management, which is best-effort |
| Queue management | Yes (Queue CRD) | None |
| Job types | VolcanoJob with plugins, supports MPI/Spark/etc. | Single CRD (TrainJob) |
| Scheduler | Custom scheduler (`volcano-scheduler`) | Uses default kube-scheduler |
| Training-specific features | Minimal — it's a generic batch scheduler | GPU prolog, auto-parallelism, checkpoint management, monitoring sidecar |

**Bottom line**: Volcano solves a real problem (gang scheduling) that this operator ignores entirely. For multi-node GPU training, gang scheduling matters — without it you risk partial allocation where some pods block resources waiting for the rest. This operator just creates a StatefulSet and hopes for the best, which works on a dedicated cluster but falls apart in shared environments.

### vs. NVIDIA NeMo Operator

[NeMo Operator](https://github.com/NVIDIA/NeMo-Framework-Launcher) (part of NVIDIA's NeMo Framework) manages training jobs specifically for NVIDIA NeMo models. It handles multi-node training with Slurm or Kubernetes.

| Aspect | NeMo Operator | TrainJob Operator |
|---|---|---|
| Framework coupling | Tightly coupled to NeMo | Framework-agnostic |
| Model support | NeMo models (GPT, Llama, T5, etc.) | Any model that works with torchrun |
| Cluster managers | Slurm + Kubernetes | Kubernetes only |
| GPU health checks | Relies on NVIDIA DCGM / GPU Operator | Built-in prolog (more opinionated) |
| Auto-parallelism | NeMo has its own parallelism auto-tuner | Built-in heuristic advisor |
| Production readiness | Yes (backed by NVIDIA) | No |

### vs. Topology-Aware Scheduling (TAS)

Topology-Aware Scheduling (TAS) is a feature in Kueue (and an area of active development in the Kubernetes scheduling ecosystem) that places pods with awareness of the physical network topology — e.g., putting all pods of a training job under the same IB switch or in the same rack to minimize communication latency.

This operator does **none of that**. It sets `nodeSelector` for GPU type but has no concept of network topology, rack placement, or NVLink/IB switch locality. For large-scale training (64+ nodes), this matters a lot — cross-rack NCCL collectives can be significantly slower than intra-rack ones.

---

## Limitations

Being straightforward about what doesn't work or isn't great:

- **No gang scheduling.** The operator creates a StatefulSet, which means pods are scheduled individually. In a shared cluster, you can get stuck with half your pods scheduled and the other half pending. This is a dealbreaker for real multi-tenant environments.
- **No elastic training.** If a node dies, the whole job fails (or checkpoints and fails). There's no automatic resizing or re-ranking. Kubeflow's ElasticPolicy or TorchElastic handle this; this operator does not.
- **No built-in queueing or quota management.** Without Kueue, jobs go straight through. The operator supports suspend-based Kueue integration (see [Kueue Integration](#kueue-integration)), but doesn't implement its own queue.
- **Auto-parallelism is heuristic-only.** The throughput model uses fixed overhead percentages that are approximations. It doesn't profile the actual workload. Think of it as a reasonable first guess, not an optimized configuration.
- **No topology-aware scheduling.** The operator doesn't know about rack layout, IB switch topology, or NVLink domains beyond a single node.
- **Checkpoint management is basic.** The operator can trigger validation jobs on checkpoint saves, but actual checkpoint writing is the training script's responsibility. The operator just manages the lifecycle around it.
- **Prolog is opinionated.** The GPU health check script is hardcoded in the prolog builder. You can't easily customize the checks or skip individual phases.
- **Built-in model/GPU registries are static.** The registries are Go maps. In production you'd back these with ConfigMaps or an external service for dynamic updates.
- **No real end-to-end test against actual GPUs.** There are envtest-based controller tests, but no integration test against a real cluster with GPU hardware.

---

## What's Missing for Production

| Gap | What you'd need |
|---|---|
| Elastic training | TorchElastic integration or a custom rendezvous backend |
| Gang scheduling | Kueue suspend integration exists, but co-scheduling needs a scheduler plugin or Volcano |
| Multi-cluster | MultiKueue or a federation layer |
| Real checkpoint storage | CSI driver for a parallel filesystem (Lustre, GPFS) or S3 integration |
| GPU topology scheduling | Custom scheduler plugin or Kueue TAS |
| Dynamic model registry | ConfigMap-backed registry with a watcher |
| Metrics pipeline | Prometheus + Grafana for the sidecar's metrics and straggler data |
| Cost-aware auto-config | Pricing API integration to optimize $/GPU-hour, not just throughput |
| RBAC / multi-tenancy | Namespace-scoped quotas, admission policies |
| CI/CD | Automated testing, image builds, Helm chart or kustomize overlays |

---

## Building

Requires Go 1.24+.

```bash
make          # fmt, vet, build → bin/trainjob-operator
make test     # run envtest-based controller tests
make run      # run locally against current kubeconfig
```

Docker:

```bash
make docker-build IMG=ghcr.io/rootfs/trainjob-operator:latest
make docker-push  IMG=ghcr.io/rootfs/trainjob-operator:latest
```

Run `make help` to see all targets.

---

## Project Structure

```
api/v1alpha1/
  types.go                       CRD types (TrainJobSpec, TrainJobStatus, phases)
  groupversion_info.go           GroupVersion, SchemeBuilder, AddToScheme
  zz_generated.deepcopy.go       DeepCopy implementations for all CRD types

cmd/
  main.go                        Entry point: manager, controller, webhooks, health checks

internal/controller/
  trainjob_controller.go         Reconciler with state-machine phase transitions
  trainjob_controller_test.go    envtest-based integration tests
  suite_test.go                  Envtest suite setup (k8sClient, manager bootstrap)
  prolog.go                      Prolog Job builder (3-phase GPU health check)
  workers.go                     Headless Service + worker StatefulSet + GPU sidecar builder
  checkpoint.go                  Checkpoint validation Job builder

internal/webhook/
  trainjob_validator.go          Validating webhook (~10 rules)
  trainjob_mutator.go            Mutating webhook (auto-parallelism, NCCL env, defaults, Kueue suspend)
  auto_config.go                 Auto-parallelism search engine (hierarchical, cached)
  model_registry.go              Built-in model architectures and GPU specs

examples/
  trainjob_sample.yaml           Sample CRs (manual, auto, Kueue, standalone, rejected configs)

Makefile                         Build, test, lint, docker targets
Dockerfile                       Multi-stage build (distroless runtime)
go.mod / go.sum                  Go module (github.com/rootfs/trainjob-operator)
```

---

## License

This project is provided as-is for educational and exploratory purposes. No warranty, no support guarantees.
