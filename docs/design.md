# Design

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

The prolog uses the actual training image, not a generic test image. This catches software stack issues (wrong CUDA version, missing libraries, etc.) in addition to hardware problems.

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

**2. No cluster-wide pod webhook.**

The GPU monitoring sidecar is now included directly in the worker StatefulSet's pod template by the reconciler. This eliminates the need for a `MutatingWebhookConfiguration` on pods entirely — no cluster-wide webhook intercepting every pod creation.

The legacy `PodSidecarInjector` is retained in the codebase for clusters that prefer webhook-based injection, but it is not registered by default.

**3. Auto-parallelism results are cached.**

The brute-force search (~microseconds) is cached by `(model, GPU type, numNodes, gpusPerNode)`. Hyperparameter sweeps that submit 50 TrainJobs with the same model and cluster shape pay the search cost once.

**4. No webhook for the prolog or worker resources.**

The prolog Job, headless Service, and worker StatefulSet are created by the reconciler (server-side), not by the user. They never go through admission webhooks.

### What we'd still want to improve

- **MutatingAdmissionPolicy for TrainJob defaults**: Once MAPs reach beta, the TrainJob defaulting (NCCL env vars, checkpoint defaults) could run as an in-process CEL policy inside kube-apiserver — no external webhook server needed for the simple cases. The auto-parallelism search would still need a webhook since it's too complex for CEL.
