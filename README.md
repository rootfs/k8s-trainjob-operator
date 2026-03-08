# TrainJob Operator

A Kubernetes operator for managing distributed GPU training jobs. It handles the full lifecycle: admission-time validation, pre-training GPU health checks, worker orchestration via StatefulSets, checkpoint management, and per-pod GPU monitoring sidecars.

The original motivation was training multimodal embedding models for [vLLM Semantic Router](https://github.com/vllm-project/semantic-router) — things like fine-tuning CLIP, SigLIP, or BGE-style encoders that power the routing layer. Managing distributed GPU training for these models on Kubernetes had enough rough edges that it made sense to build an operator around it. The scope grew from there into a more general-purpose training job operator.

Built with [controller-runtime](https://github.com/kubernetes-sigs/controller-runtime) and [Kubebuilder](https://book.kubebuilder.io/) conventions.

> **Disclaimer**: This is a hobby / exploration project. It is **not** production-ready and is not affiliated with any company. The code was written to learn and experiment with Kubernetes operator patterns for GPU training workloads. There are known gaps (see [Limitations](docs/comparison.md#limitations)). Use at your own risk.

---

## Problem Statement

Kubernetes gives you primitives — Pods, Jobs, StatefulSets — but distributed GPU training needs a lot of domain-specific glue between "I want to train this model" and "training completed successfully." This operator tries to own that glue.

The pain points it targets:

- **Configuration correctness** — Getting TP/PP/FSDP/CP right for a given model and hardware combo is genuinely hard. The validating webhook and auto-parallelism advisor catch bad configs at admission time — fail fast, not fail late.
- **Hardware reliability** — At scale, bad GPUs, degraded NVLink, flaky InfiniBand ports are expected. The prolog is a pre-flight check that runs before training starts.
- **Runtime observability** — The GPU monitoring sidecar watches for stragglers, hardware anomalies, and step-time regressions while training runs.
- **Checkpoint lifecycle** — The operator manages the save/validate/retain cycle and protects the last checkpoint on deletion.
- **Model-infra co-design** — The auto-parallelism advisor matches model architecture to hardware, finding a good parallelism config automatically.

This is not a scheduler (that's Kueue/Volcano), not a training framework (that's PyTorch/NeMo), and not a cluster provisioner. It's the **operational layer in between**.

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
        ├─ NCCL env vars              ├─ GPU memory estimation
        └─ Default configs            └─ Head divisibility by TP
                 │                             │
                 └─────────────┬───────────────┘
                               ▼
                 ┌─────────────────────────────┐
                 │   Reconciler State Machine   │
                 │                              │
                 │   Suspended ──► Pending       │
                 │     └─► PrologRunning        │
                 │           ├─► Running        │
                 │           │    ├─► Succeeded │
                 │           │    ├─► Checkpoint │
                 │           │    └─► Failed    │
                 │           └─► PrologFailed   │
                 └─────────────────────────────┘
```

---

## Documentation

| Document | What it covers |
|---|---|
| [CRD Reference](docs/crd.md) | TrainJob spec, ModelArchSpec, CheckpointSpec, status fields, YAML examples |
| [Design](docs/design.md) | Workflow (admission → prolog → workers → monitoring → deletion), auto-parallelism advisor (hierarchical search, pruning, caching), webhook efficiency |
| [Kueue Integration](docs/kueue.md) | Suspend-based admission, child resource isolation, standalone mode |
| [Comparison & Limitations](docs/comparison.md) | vs. Kubeflow, Kueue, Volcano, NeMo, TAS; known limitations; production gaps |
| [Multi-Agent System](agents/README.md) | 8 specialized agents, interaction diagrams, coordination protocol, model-install agent, run→observe→eval→iterate harness engineering pipeline |

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

Agents:

```bash
make agent-build                   # build agent binary
make agent-run ROLE=model-builder  # dry-run an agent locally
make agent-eval                    # evaluate agent traces
make agent-pipeline ROLE=sre       # full run → eval → iterate
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

docs/
  crd.md                         CRD reference and examples
  design.md                      Workflow, auto-parallelism, webhook efficiency
  kueue.md                       Kueue integration details
  comparison.md                  Framework comparison and limitations

agents/                          Multi-agent system (see agents/README.md)
examples/                        Sample TrainJob CRs

Makefile                         Build, test, lint, docker, agent targets
Dockerfile                       Multi-stage build (distroless runtime)
go.mod / go.sum                  Go module (github.com/rootfs/trainjob-operator)
```

---

## License

This project is provided as-is for educational and exploratory purposes. No warranty, no support guarantees.
