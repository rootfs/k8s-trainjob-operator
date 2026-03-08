# Multi-Agent System

The project includes an experimental multi-agent system where specialized AI agents collaborate on operator development via git-based coordination. Each agent is a Go binary that connects to a vLLM inference server (OpenAI-compatible API with tool calling), reads its role-specific `SKILL.md` instructions, claims tasks from a shared `AGENCY.md` file, and pushes changes to feature branches.

## Agents

| Agent | Scope | What it does |
|---|---|---|
| `model-builder` | `model_registry.go`, CRD types | Populate and maintain model architecture definitions |
| `model-trainer` | `auto_config.go`, mutating webhook | Improve auto-parallelism search and training config |
| `infra` | CRDs, resource builders, Kueue | StatefulSet/Job builders, Kueue integration |
| `ops` | Reconciler, webhooks, checkpoints | State machine logic, webhook validation rules |
| `sre` | Workers, prolog | GPU health checks, monitoring sidecar, DCGM metrics |
| `cicd` | Makefile, Dockerfile, tests | Build pipeline, test coverage, GitHub Actions |
| `model-eval` | Checkpoint, reconciler, types | Post-training evaluation benchmarks and regression detection |
| `model-install` | Checkpoint, reconciler, vLLM | Checkpoint conversion, registry push, serving config generation |

## Agent Iteration Loop

Each agent run follows the same execution cycle. The agent binary is stateless — it clones fresh, does one unit of work, and exits. State is carried entirely in git.

```
┌──────────────────────────────────────────────────────────────────┐
│                        Agent Run (K8s Job)                       │
│                                                                  │
│  ┌──────────┐    ┌──────────────┐    ┌────────────┐             │
│  │ git clone├───►│ Read AGENCY  ├───►│ Read SKILL │             │
│  │ (depth=1)│    │    .md       │    │    .md     │             │
│  └──────────┘    └──────────────┘    └─────┬──────┘             │
│                                            │                     │
│                                            ▼                     │
│                          ┌─────────────────────────────┐         │
│                          │ Build system prompt:         │         │
│                          │  • Role description          │         │
│                          │  • SKILL.md instructions     │         │
│                          │  • AGENCY.md task board      │         │
│                          │  • Watched file paths        │         │
│                          └──────────────┬──────────────┘         │
│                                         │                        │
│                       git checkout -b agent/<role>/<ts>           │
│                                         │                        │
│                                         ▼                        │
│  ┌─────────── Agent Loop (up to 20 turns) ──────────────────┐   │
│  │                                                           │   │
│  │   Agent ──POST──► vLLM (/v1/chat/completions)             │   │
│  │                    │                                      │   │
│  │          ┌─────────┴──────────┐                           │   │
│  │          ▼                    ▼                            │   │
│  │     tool_calls            text response                   │   │
│  │     ┌────────────┐        (thinking / done)               │   │
│  │     │ read_file  │                                        │   │
│  │     │ edit_file  │                                        │   │
│  │     │ write_file │   Execute ──► Result appended          │   │
│  │     │ run_command│   locally      to messages             │   │
│  │     │ list_files │                    │                    │   │
│  │     │ search_file│                    │                    │   │
│  │     │ git_diff   │                    ▼                    │   │
│  │     │ git_commit │              Next turn ───►            │   │
│  │     │ done()     │                                        │   │
│  │     └────────────┘                                        │   │
│  │                                                           │   │
│  │   done() triggers:  git push ──► feature branch           │   │
│  └───────────────────────────────────────────────────────────┘   │
│                                                                  │
│  Exit: branch ready for human review / PR                        │
└──────────────────────────────────────────────────────────────────┘
```

## Agent Interaction Map

Agents don't talk to each other directly. They interact through shared files in the git repository. `AGENCY.md` is the task board — agents claim work, mark completion, and leave notes. The actual code artifacts (Go files, YAML manifests) are the shared state.

```
                              AGENCY.md
                            (task board)
                     ┌──── claim / complete ────┐
                     │                          │
        ┌────────────┴───────────────────────────┴────────────┐
        │                    Git Repository                    │
        │                                                      │
        │   api/v1alpha1/types.go ◄────────┐                   │
        │        ▲    ▲    ▲               │                   │
        │        │    │    │               │                   │
        │   model-    │   infra         model-                 │
        │   builder   │                 eval                   │
        │             │                   │                    │
        │         model-              model-                   │
        │         install             install                  │
        │             │                   │                    │
        │             ▼                   ▼                    │
        │   internal/controller/ ◄─── ops                      │
        │   ├── trainjob_controller.go                         │
        │   ├── workers.go ◄───────── sre                      │
        │   ├── prolog.go ◄────────── sre                      │
        │   ├── checkpoint.go ◄────── model-eval, model-install│
        │   └── install.go ◄───────── model-install            │
        │                                                      │
        │   internal/webhook/ ◄─── model-trainer, ops          │
        │   ├── auto_config.go                                 │
        │   ├── model_registry.go ◄── model-builder            │
        │   ├── trainjob_mutator.go                            │
        │   └── trainjob_validator.go                          │
        │                                                      │
        │   Makefile, Dockerfile ◄─── cicd                     │
        │   .github/workflows/  ◄──── cicd                     │
        └──────────────────────────────────────────────────────┘
                     │
                     │  Each agent pushes to its own branch:
                     │  agent/model-builder/initial-model-registry
                     │  agent/sre/prolog-ib-detection
                     │  agent/model-install/install-conversion-job
                     │  ...
                     ▼
              Human reviews PR
              Merges to main
```

## End-to-End Pipeline: Train → Eval → Install → Serve

The operator manages the training lifecycle. The agents extend the pipeline from training completion through evaluation, model conversion, and deployment config generation.

```
  ┌─ TrainJob Operator (reconciler) ────────────────────────────────────┐
  │                                                                     │
  │  Suspended ──► Prolog ──► Running ──► Checkpointing ──► Succeeded  │
  │  (Kueue)      (GPU       (StatefulSet  (periodic       (training   │
  │               health)     + sidecar)    save/validate)  complete)  │
  │                                                                     │
  └─────────────────────────────────┬───────────────────────────────────┘
                                    │
                          checkpoint on PVC/S3
                                    │
                                    ▼
  ┌─ model-eval agent ─────────────────────────────────────────────────┐
  │                                                                     │
  │  Load checkpoint ──► Run benchmark suite ──► Compare to baseline   │
  │  (MTEB, BEIR,        (embedding quality,     (regression check:    │
  │   custom domain)       retrieval metrics)      ΔRecall, ΔNDCG)     │
  │                                                                     │
  │  Output: eval-results.json on branch, pass/fail decision           │
  └─────────────────────────────────┬───────────────────────────────────┘
                                    │
                              eval passed ✓
                                    │
                                    ▼
  ┌─ model-install agent ──────────────────────────────────────────────┐
  │                                                                     │
  │  ┌────────────────┐    ┌─────────────────┐    ┌──────────────────┐ │
  │  │ 1. Convert     │    │ 2. Registry     │    │ 3. Serving       │ │
  │  │                │    │    Push          │    │    Config        │ │
  │  │ Sharded .pt ──►│───►│                 │───►│                  │ │
  │  │ safetensors    │    │ OCI / HF / S3   │    │ vLLM Deployment  │ │
  │  │ (+ quantize)   │    │ artifact        │    │ patch + router   │ │
  │  │                │    │                 │    │ ConfigMap update  │ │
  │  └────────────────┘    └─────────────────┘    └──────────────────┘ │
  │                                                                     │
  │  Output: deployment manifests on branch, NOT applied directly      │
  └─────────────────────────────────┬───────────────────────────────────┘
                                    │
                              PR / GitOps sync
                                    │
                                    ▼
  ┌─ Serving Cluster (human / ArgoCD / Flux) ──────────────────────────┐
  │                                                                     │
  │  vLLM Deployment ◄── updated model path                            │
  │  Semantic Router ◄── updated routing ConfigMap                      │
  │                                                                     │
  │  Optional: canary deployment (low replica) ──► promote / rollback  │
  └─────────────────────────────────────────────────────────────────────┘
```

## Coordination Protocol

Agents coordinate through `AGENCY.md`, a git-tracked file that acts as a task board with ownership, status, dependencies, and notes.

```
Protocol:
                    ┌──────────┐
                    │  Agent   │
                    │  starts  │
                    └────┬─────┘
                         │
                    git pull main
                         │
                         ▼
                 ┌───────────────┐     blocked by
                 │ Find unclaimed├────dependency───► skip, add note,
                 │ task for role │                   try next task
                 └───────┬───────┘
                         │
                  Set Owner + Status = in_progress
                  Commit AGENCY.md to branch
                         │
                         ▼
                 ┌───────────────┐
                 │ Do the work   │
                 │ (agent loop)  │
                 └───────┬───────┘
                         │
                  Set Status = proposed
                  Push branch
                         │
                         ▼
                 ┌───────────────┐
                 │ Human reviews │
                 │ and merges PR │
                 └───────────────┘
```

## Model Install Agent

The model-install agent owns the last mile between training and serving. After a model passes evaluation, it needs to be converted from distributed training checkpoints (sharded PyTorch files) to a serving format (HuggingFace safetensors for vLLM), pushed to a model registry (OCI, HuggingFace Hub, or S3), and deployed to the inference engine.

The agent generates deployment artifacts (conversion Jobs, registry push scripts, vLLM Deployment patches, semantic router ConfigMap updates) and pushes them to a feature branch. It does **not** apply changes to the serving cluster directly — the actual rollout is a human or GitOps decision.

This is a standalone agent rather than a reconciler phase because:
- Conversion/push are one-shot operations, not lifecycle phases that need continuous reconciliation
- The serving cluster may be different from the training cluster
- Deployment decisions (canary percentages, rollback triggers) are policy decisions that belong in a CD pipeline, not an operator

## Run → Observe → Eval → Iterate Pipeline (Harness Engineering)

The agent system includes a built-in feedback loop — an approach known as [harness engineering](https://blog.langchain.com/improving-deep-agents-with-harness-engineering/). The harness (SKILL.md + WatchPaths + protocol + verification gates) constrains and guides each agent. The eval + iterate pipeline tightens the harness over time based on observed failures.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│   ┌───────┐     ┌─────────┐     ┌──────┐     ┌─────────┐              │
│   │  RUN  │────►│ OBSERVE │────►│ EVAL │────►│ ITERATE │──┐           │
│   └───────┘     └─────────┘     └──────┘     └─────────┘  │           │
│       ▲                                                     │           │
│       │              improved SKILL.md                      │           │
│       └─────────────────────────────────────────────────────┘           │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

**Phase 1: Run** — agent executes its task, producing code changes on a feature branch. Every tool call is recorded with timing and error status.

**Phase 2: Observe** — the trace captures the full execution trajectory as JSON in `traces/<role>_<run_id>.json`. Includes tool calls, file access patterns, build/vet status, and exit reason.

**Phase 3: Eval** — seven deterministic scorers + optional LLM-as-judge evaluate each trace:

| Scorer | Weight | What it catches |
|---|---|---|
| `build_gate` | 3.0 | Code that doesn't compile |
| `vet_gate` | 2.0 | go vet violations |
| `scope_adherence` | 2.5 | Files modified outside the role's WatchPaths |
| `protocol_compliance` | 1.5 | Skipped reading AGENCY.md, edited without reading, no build check |
| `efficiency` | 1.0 | Used too many turns relative to max |
| `tool_error_rate` | 1.0 | High percentage of failed tool calls |
| `completion_signal` | 1.5 | Didn't call done(), hit max_turns, errored out |
| `golden_match` | 2.0 | Expected files, required tools, turn budget (if golden case exists) |
| `llm_judge` | 2.0 | Task accomplishment, code quality, engineering practices (if vLLM available) |

A trace passes if: overall weighted score >= 0.6 AND no critical failure (build_gate=0 or scope_adherence=0).

**Phase 4: Iterate** — reads all eval reports and traces, extracts recurring failure patterns (analysis paralysis, repeated tool errors, budget exhaustion, scope violations), and proposes targeted SKILL.md edits. With `ITERATE_APPLY=true`, high-priority proposals are automatically written into SKILL.md files so the next agent run gets improved instructions.

## Running

```bash
# Build the agent binary
make agent-build

# --- Agent execution ---
AGENT_ROLE=model-install AGENT_REPO_URL=https://github.com/rootfs/trainjob-operator \
  VLLM_ENDPOINT=http://localhost:8000/v1 make agent-run

# --- Eval pipeline ---
make agent-eval                                              # deterministic scorers only
VLLM_ENDPOINT=http://localhost:8000/v1 make agent-eval       # + LLM-as-judge
VLLM_ENDPOINT=http://localhost:8000/v1 make agent-iterate    # propose SKILL.md improvements
make agent-iterate-apply                                     # auto-apply high-priority proposals

# --- Full pipeline ---
make agent-pipeline ROLE=model-builder                       # run → eval → iterate

# --- K8s deployment ---
make agent-deploy           # deploy vLLM + RBAC + CronJobs
make agent-job ROLE=sre     # one-shot agent Job
```
