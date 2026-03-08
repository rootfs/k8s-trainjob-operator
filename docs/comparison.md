# Comparison with Existing Frameworks

This section tries to be fair about what this project does and doesn't do relative to established tools.

## vs. Kubeflow Training Operator

[Kubeflow Training Operator](https://github.com/kubeflow/training-operator) is the standard way to run distributed training on Kubernetes. It provides CRDs like `PyTorchJob`, `TFJob`, `MPIJob`, etc.

| Aspect | Kubeflow Training Operator | TrainJob Operator |
|---|---|---|
| Maturity | Production-grade, widely adopted, CNCF project | Hobby project, not battle-tested |
| Framework support | PyTorch, TensorFlow, MPI, XGBoost, PaddlePaddle, JAX | Framework-agnostic (anything torchrun-compatible) |
| Elastic training | Supports TorchElastic via `ElasticPolicy` | Not supported |
| GPU health checks | Not built-in | Built-in prolog Job (hardware, kernel, interconnect) |
| Auto-parallelism | Not built-in | Built-in heuristic advisor (with caveats) |
| Checkpoint management | Not built-in (left to the training script) | Operator-managed: periodic save, validation, retention |
| Admission validation | Basic (schema validation) | Domain-specific rules (TP/PP constraints, memory estimation, FP8 GPU checks) |
| GPU monitoring sidecar | Not built-in | Injected via pod webhook (DCGM, straggler detection, anomaly watching) |
| Gang scheduling | Integrates with Volcano / Kueue | Not supported (relies on StatefulSet parallel pod management) |
| Community & support | Large community, good docs | Just me |

**Bottom line**: Kubeflow Training Operator is the right choice for production. This project explores ideas that Kubeflow doesn't cover out of the box (prolog checks, auto-parallelism, monitoring sidecar), but it lacks the breadth, maturity, and community that Kubeflow has.

## vs. Kueue

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

**Bottom line**: Kueue and a training operator are complementary, not competing. In a real setup you'd use Kueue for admission control and scheduling, and something like this operator (or Kubeflow) for the training-specific lifecycle. This project supports suspend-based Kueue integration — see [Kueue Integration](kueue.md).

## vs. Volcano

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

## vs. NVIDIA NeMo Operator

[NeMo Operator](https://github.com/NVIDIA/NeMo-Framework-Launcher) (part of NVIDIA's NeMo Framework) manages training jobs specifically for NVIDIA NeMo models. It handles multi-node training with Slurm or Kubernetes.

| Aspect | NeMo Operator | TrainJob Operator |
|---|---|---|
| Framework coupling | Tightly coupled to NeMo | Framework-agnostic |
| Model support | NeMo models (GPT, Llama, T5, etc.) | Any model that works with torchrun |
| Cluster managers | Slurm + Kubernetes | Kubernetes only |
| GPU health checks | Relies on NVIDIA DCGM / GPU Operator | Built-in prolog (more opinionated) |
| Auto-parallelism | NeMo has its own parallelism auto-tuner | Built-in heuristic advisor |
| Production readiness | Yes (backed by NVIDIA) | No |

## vs. Topology-Aware Scheduling (TAS)

Topology-Aware Scheduling (TAS) is a feature in Kueue (and an area of active development in the Kubernetes scheduling ecosystem) that places pods with awareness of the physical network topology — e.g., putting all pods of a training job under the same IB switch or in the same rack to minimize communication latency.

This operator does **none of that**. It sets `nodeSelector` for GPU type but has no concept of network topology, rack placement, or NVLink/IB switch locality. For large-scale training (64+ nodes), this matters a lot — cross-rack NCCL collectives can be significantly slower than intra-rack ones.

---

## Limitations

Being straightforward about what doesn't work or isn't great:

- **No gang scheduling.** The operator creates a StatefulSet, which means pods are scheduled individually. In a shared cluster, you can get stuck with half your pods scheduled and the other half pending. This is a dealbreaker for real multi-tenant environments.
- **No elastic training.** If a node dies, the whole job fails (or checkpoints and fails). There's no automatic resizing or re-ranking. Kubeflow's ElasticPolicy or TorchElastic handle this; this operator does not.
- **No built-in queueing or quota management.** Without Kueue, jobs go straight through. The operator supports suspend-based Kueue integration (see [Kueue Integration](kueue.md)), but doesn't implement its own queue.
- **Auto-parallelism is heuristic-only.** The throughput model uses fixed overhead percentages that are approximations. It doesn't profile the actual workload. Think of it as a reasonable first guess, not an optimized configuration.
- **No topology-aware scheduling.** The operator doesn't know about rack layout, IB switch topology, or NVLink domains beyond a single node.
- **Checkpoint management is basic.** The operator can trigger validation jobs on checkpoint saves, but actual checkpoint writing is the training script's responsibility. The operator just manages the lifecycle around it.
- **Prolog is opinionated.** The GPU health check script is hardcoded in the prolog builder. You can't easily customize the checks or skip individual phases.
- **Built-in model/GPU registries are static.** The registries are Go maps. In production you'd back these with ConfigMaps or an external service for dynamic updates.
- **No real end-to-end test against actual GPUs.** There are envtest-based controller tests, but no integration test against a real cluster with GPU hardware.

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
