# Complete TrainJob Operator — Full Implementation

A production-grade K8s operator for distributed GPU training jobs, written end-to-end
so you can see every piece and how they connect. This implements the patterns from
the `k8s_operator_cheat_sheet.md` as real, compilable Go code.

---

## Architecture Overview

```
                        User creates TrainJob CR
                                  │
                    ┌─────────────┴─────────────┐
                    ▼                             ▼
           Mutating Webhook              Validating Webhook
           (trainjob_mutator.go)         (trainjob_validator.go)
           │                              │
           ├─ AUTO-PARALLELISM ADVISOR    ├─ TP ≤ GPUs/node?
           │  (if autoParallelism=true)   ├─ FP8 only on Hopper+?
           │  Derives TP/PP/FSDP/CP/      ├─ Checkpoint for long jobs?
           │  batch/precision from         ├─ Total GPUs % (TP×PP×CP) = 0?
           │  ModelSpec + GPU hardware     ├─ GPU memory estimation
           │                              ├─ NCCL bandwidth feasibility
           ├─ Inject NCCL env vars        ├─ MHA head divisibility by TP
           │  based on GPU type           └─ KV head divisibility by TP
           ├─ Default precision=bf16
           ├─ Default CP=1
           └─ Default checkpoint config
                    │                             │
                    └─────────────┬───────────────┘
                                  │ (admitted)
                                  ▼
                          API Server persists CR
                                  │
                                  ▼
                    ┌─────────────────────────────┐
                    │   TrainJob Reconciler        │
                    │   (trainjob_controller.go)   │
                    │                              │
                    │   State Machine:             │
                    │   ┌────────┐                 │
                    │   │Pending │                 │
                    │   └───┬────┘                 │
                    │       │                      │
                    │       ▼                      │
                    │   ┌────────────┐             │
                    │   │PrologRunning│ ──fail──┐  │
                    │   └─────┬──────┘          │  │
                    │         │pass              │  │
                    │         ▼                  │  │
                    │   ┌────────────┐     ┌─────▼┐│
                    │   │PrologPassed│     │Failed ││
                    │   └─────┬──────┘     └──────┘│
                    │         │creates:            │
                    │         ├─ Headless Service   │
                    │         └─ Worker StatefulSet │
                    │         ▼                    │
                    │   ┌────────┐                 │
                    │   │Running │ ──fail──────────┤
                    │   └───┬────┘                 │
                    │       │all pods succeed      │
                    │       ▼                      │
                    │   ┌──────────┐               │
                    │   │Succeeded │               │
                    │   └──────────┘               │
                    └─────────────────────────────┘

                    Pod creation by StatefulSet
                                  │
                                  ▼
                    ┌─────────────────────────────┐
                    │   Pod Sidecar Webhook        │
                    │   (trainjob_mutator.go)      │
                    │                              │
                    │   Injects gpu-monitor agent: │
                    │   ┌─ DCGM Exporter (:9400)  │
                    │   ├─ HW Telemetry Collector  │
                    │   ├─ Kernel Perf Monitor     │
                    │   ├─ Straggler Detector      │
                    │   └─ Anomaly Watchdog        │
                    │      (ECC, thermal, OOM,     │
                    │       XID, NCCL timeout)     │
                    │   Managed by supervisor loop │
                    └─────────────────────────────┘
```

---

## File Map

```
api/v1alpha1/
  types.go                  CRD types: TrainJobSpec, TrainJobStatus, phases, conditions
                            Kubebuilder markers for printer columns, status subresource

internal/controller/
  trainjob_controller.go    Main reconciler — state machine with 8 phase handlers
                            Finalizer for checkpoint preservation on deletion
                            Owner references: TrainJob owns Jobs, StatefulSet, Service
                            RBAC markers for all child resource types

  prolog.go                 Builds the 3-phase prolog validation Job
                            Indexed completion (one pod per node, all parallel)
                            Uses the TRAINING IMAGE (not a generic image) so it
                            validates the exact software stack the job will use.

                            Phase 1 — Hardware Health:
                              nvidia-smi GPU count, ECC errors, DCGM diag -r 2,
                              clock throttle check, PCIe link width/speed, IB port status

                            Phase 2 — Kernel Validation:
                              compute-sanitizer memcheck on CUDA matmul,
                              torch.compile warmup (verifies Triton codegen on this HW),
                              precision smoke test (bf16/fp8 matmul vs fp32 reference),
                              cross-GPU consistency check (same matmul → same result,
                              catches per-GPU SDC)

                            Phase 3 — Interconnect Bandwidth:
                              P2P bandwidth matrix (all GPU pairs on node),
                              detects degraded NVLink / asymmetric link failures

  workers.go                Builds the headless Service + worker StatefulSet
                            StatefulSet (not Deployment) for stable rank assignment
                            ParallelPodManagement for simultaneous startup
                            Env injection: MASTER_ADDR, WORLD_SIZE, RANK via pod ordinal
                            /dev/shm volume for NCCL shared memory, 5-min grace period

  checkpoint.go             Builds the checkpoint validation Job
                            Python script: NaN/Inf scan, shard count, loss range, grad norm

internal/webhook/
  trainjob_validator.go     Validating webhook — 10 rules:
                            TP ≤ GPUs/node, total_GPUs % (TP×PP×CP) = 0,
                            FP8 only Hopper+, checkpoint required for long jobs,
                            numNodes > 0, valid GPU counts, image required,
                            no parallelism change mid-training,
                            GPU memory estimation, NCCL bandwidth feasibility

  trainjob_mutator.go       Mutating webhook — TrainJob defaults:
                            ★ AUTO-PARALLELISM ADVISOR (autoParallelism=true):
                              brute-force search ~1600 configs to find optimal
                              TP/PP/FSDP/CP/batch/precision for model+hardware
                            NCCL env vars per GPU type (H100/A100/L40),
                            default precision=bf16, default CP=1,
                            default checkpoint config

                            Pod sidecar injector — 5-subsystem observability agent:
                              1. DCGM Exporter (Prometheus GPU metrics)
                              2. HW Telemetry Collector (PCIe/NVLink errors, IB,
                                 CPU/mem/IO, power trends, throttle reasons)
                              3. Kernel Perf Monitor (step-time rolling baseline,
                                 regression detection at 20% threshold)
                              4. Straggler Detector (self-check vs baseline,
                                 heartbeat for cluster-wide aggregation)
                              5. Anomaly Watchdog (ECC accumulation → emergency ckpt,
                                 thermal runaway, OOM proximity, XID errors from dmesg,
                                 NCCL flight recorder dump detection)
                            Supervisor loop restarts crashed subsystems

  auto_config.go            Auto-parallelism engine:
                            Enumerates TP/PP/CP/batch/act-ckpt/precision combos,
                            estimates memory + throughput for each,
                            picks highest-throughput config that fits in GPU memory.
                            Heuristic throughput model considers TP/PP/FSDP overhead,
                            pipeline bubbles, activation recompute, FP8 speedup.

  model_registry.go         Built-in model architectures (Llama 3 family, Mixtral,
                            Mistral, GPT-4-scale) and GPU specs (H100/H200/A100/L40).
                            In production, backed by ConfigMap for dynamic updates.

cmd/
  main.go                   Entry point: creates manager, registers controller,
                            registers all 3 webhooks, health checks, leader election

examples/
  trainjob_sample.yaml      Sample CRs: 70B pretrain, 8B finetune,
                            auto-parallelism (70B, custom 13B, 405B),
                            rejected examples, mutator behavior

internal/controller/
  trainjob_controller_test.go   envtest-based tests:
                                skip-prolog happy path, prolog pass/fail,
                                finalizer checkpoint preservation
```

---

## How the Pieces Connect — Walkthrough

### Step 1: User submits TrainJob CR

```bash
kubectl apply -f examples/trainjob_sample.yaml
```

### Step 2: Mutating webhook fires (trainjob_mutator.go)

Before the CR is persisted, the mutating webhook runs in this order:

**2a. Auto-parallelism advisor** (if `spec.autoParallelism: true`):
- Resolves model architecture from `spec.modelSpec` or `spec.model` name → `ModelRegistry`
- Looks up GPU hardware from `nodeSelector` → `GPURegistry`
- Calls `AutoConfigureParallelism()` which:
  - Enumerates all valid TP (power-of-2, divides heads), PP (divides layers), CP, micro-batch combos
  - For each combo: estimates per-GPU memory; discards configs that exceed GPU memory
  - Estimates relative throughput (accounts for TP overhead, PP bubble, FSDP IB cost, FP8 speedup)
  - Returns the highest-throughput feasible config (~1,600 combos, microsecond search)
- Writes `tpDegree`, `ppDegree`, `cpDegree`, `precision`, `modelSpec.microBatchSize`,
  `modelSpec.activationCheckpointing` back into the spec
- Annotates the CR with `training.vsr.dev/auto-parallelism-config` and
  `training.vsr.dev/auto-parallelism-reason` for observability

**2b. Defaults**:
- Sets `precision: bf16` if empty
- Sets `cpDegree: 1` if not set
- Sets `checkpoint.intervalMinutes: 30` if not specified
- Sets `checkpoint.retainCount: 3` if not specified

**2c. NCCL env vars** injected based on `nodeSelector["nvidia.com/gpu.product"]`:
- H100 → 9 NCCL vars (IB, HCA, cross-NIC, etc.)
- A100 → 5 NCCL vars
- L40/L4 → 3 NCCL vars (Socket-based, no IB)

### Step 3: Validating webhook fires (trainjob_validator.go)

After mutation, the validator checks all 10 rules. If any fail, the create
is **rejected** with a specific error message the user sees immediately.

### Step 4: Reconciler picks up the new CR

The reconciler's state machine begins:

**Phase: Pending**
- If `skipProlog: true` → transition to PrologPassed
- Otherwise → create prolog Job (one pod per node, indexed completion)
- Transition to PrologRunning

**Phase: PrologRunning**
- Poll the prolog Job every 15 seconds
- The prolog runs **3 phases per node** using the training image:
  - **Phase 1 — Hardware Health**: GPU count, ECC errors, DCGM diag, clock throttling,
    PCIe link width/speed, InfiniBand ports
  - **Phase 2 — Kernel Validation**: `compute-sanitizer` memcheck on a CUDA matmul,
    `torch.compile` warmup (verifies Triton codegen works on this exact hardware),
    precision smoke test (bf16/fp8 matmul vs fp32 reference within tolerance),
    cross-GPU consistency check (same matmul → same result on every GPU, catches per-GPU SDC)
  - **Phase 3 — Interconnect Bandwidth**: P2P bandwidth matrix (all GPU pairs),
    flags degraded NVLink or asymmetric link failures
- If Job succeeds → PrologPassed
- If Job fails → PrologFailed → Failed (with specific failure message from prolog)

**Phase: PrologPassed**
- Create headless Service (for `torch.distributed` DNS rendezvous)
- Create worker StatefulSet (NumNodes replicas, each requesting GPUsPerNode GPUs)
- Set `status.startTime`
- Transition to Running

**Phase: Running**
- Poll StatefulSet every 60 seconds
- Update `status.readyWorkers`
- Check for failed pods → Checkpointing (if enabled) or Failed
- Check for all succeeded pods → Succeeded
- Check max runtime exceeded → Failed

### Step 5: StatefulSet creates worker Pods

When each Pod is created, the **pod sidecar webhook** injects a comprehensive
observability agent (`gpu-monitor` container) with 5 subsystems:

| # | Subsystem | What It Does | Outputs |
|---|---|---|---|
| 1 | **DCGM Exporter** | GPU metrics to Prometheus (:9400) | Standard DCGM metrics |
| 2 | **HW Telemetry Collector** | PCIe errors, NVLink CRC/replay counters, IB port stats, CPU/mem/IO, power trends, throttle reasons | `hw_telemetry.jsonl` |
| 3 | **Kernel Perf Monitor** | Reads step times from shared memory, builds rolling baseline, detects regressions (>20% slower than baseline) | `kernel_perf.jsonl` |
| 4 | **Straggler Detector** | Compares this worker's step time vs. baseline, publishes heartbeat for cluster-wide aggregation | `straggler.jsonl`, `straggler_heartbeat.json` |
| 5 | **Anomaly Watchdog** | ECC accumulation → USR1 signal for emergency checkpoint, thermal runaway, OOM proximity, XID errors from dmesg, NCCL flight recorder dump detection | `anomalies.jsonl` |

A **supervisor loop** monitors all subsystems and restarts any that crash.
The webhook also adds Prometheus scrape annotations and a health endpoint (:9401).

### Step 6: On deletion

The **finalizer** (`training.vsr.dev/checkpoint-protection`) ensures:
- Last checkpoint path is logged/preserved
- Only then removes the finalizer, allowing K8s garbage collection to clean up
  all child resources (Jobs, StatefulSet, Service) via owner references

---

## Patterns Used (from cheat sheet)

| Pattern | Where Used |
|---|---|
| State Machine (Pattern 3) | Main reconciler — 8 phases |
| Create-If-Not-Exists (Pattern 1) | Prolog Job, headless Service, worker StatefulSet |
| Watch-and-React (Pattern 4) | Monitoring prolog Job and worker pods |
| Owner References | TrainJob → Jobs, StatefulSet, Service (auto-cleanup) |
| Finalizer | Checkpoint preservation before deletion |
| Status Conditions | Phase transitions, AllWorkersReady, etc. |
| Validating Webhook | 8 validation rules |
| Mutating Webhook | Auto-parallelism advisor, NCCL injection, defaults, 5-subsystem sidecar agent |
| Indexed Job | Prolog: one completion per node |
| envtest | Integration tests with real API server |

---

## Auto-Parallelism: How It Works

The auto-parallelism advisor is the operator's highest-value feature. It turns
"here's my model and my cluster" into "here's your optimal parallelism config."

```
User provides:                    Webhook derives:
┌──────────────┐                 ┌───────────────────────┐
│ model name   │  ─── lookup ──▶ │ TP=8  PP=4  FSDP=8   │
│   OR         │   ModelRegistry │ CP=1  micro_batch=2   │
│ modelSpec    │                 │ act_ckpt=true         │
│              │  ─── lookup ──▶ │ precision=bf16        │
│ numNodes=32  │   GPURegistry   │                       │
│ gpusPerNode=8│                 │ score=2048.7          │
│ gpu=H100     │                 │ memory=68.3/72.0 GB   │
└──────────────┘                 └───────────────────────┘
                  brute-force
                  ~1600 combos
                  in microseconds
```

**Search space constraints (why brute-force works):**
- TP: {1, 2, 4, 8} — must be power-of-2, divide heads AND KV-heads, ≤ GPUs/node → **4** values
- PP: divisors of num_layers ≤ num_nodes → **~10** values
- CP: {1, 2, 4} — only useful for seq_len ≥ 32k → **3** values
- Micro-batch: {1, 2, 4, 8} → **4** values
- Activation checkpointing: {true, false} → **2** values
- Precision: {bf16, fp8} (if Hopper+) → **2** values
- **Total: 4 × 10 × 3 × 4 × 2 × 2 = 1,920 (upper bound)**

**Throughput heuristic considers:**
- TP overhead: AllReduce per layer on NVLink (~5% at TP=8)
- PP overhead: pipeline bubble = (PP-1)/(PP+microbatches-1) with 1F1B schedule
- FSDP overhead: AllGather/ReduceScatter on IB (70% overlap with FSDP2 prefetch)
- Activation checkpointing: ~33% more compute (recompute forward in backward)
- FP8 speedup: ~1.7× on Hopper hardware
- Micro-batch size: larger → better GPU utilization

**Why this matters at scale:**
- Manually tuning parallelism for Llama-3-405B takes senior engineers hours of trial-and-error
- A misconfigured TP/PP can waste thousands of GPU-hours before OOM or slowdown is detected
- The webhook catches misconfigs at submission time, not after 30 minutes of scheduling + prolog

---

## What's NOT in This Example (Production Additions)

| Missing | Why | What You'd Add |
|---|---|---|
| TorchElastic integration | Requires custom rendezvous backend | Replace StatefulSet with elastic ReplicaSet or MPIJob |
| Straggler detection logic | Requires metrics pipeline (Prometheus) | Separate controller watching step-time metrics |
| Kueue integration | Requires Kueue CRDs | Add `kueue.x-k8s.io/queue-name` label, wait for admission |
| Multi-cluster | Requires federation layer | MultiKueue or Admiralty dispatch |
| Real checkpoint storage | Needs S3/GCS/Lustre integration | Replace PVC with CSI driver for parallel filesystem |
| GPU topology scheduling | Needs custom scheduler plugin | NVLink-topology-aware plugin (see operator cheat sheet §11) |
| ConfigMap-backed model registry | Requires ConfigMap controller | Replace `ModelRegistry` map with ConfigMap watcher for dynamic model addition |
| Cost-based auto-config | Requires pricing API | Weight throughput score by $/GPU-hour for cost-optimal configs |
