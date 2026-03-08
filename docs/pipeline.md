# ModelPipeline: Continuous Model Lifecycle

A `ModelPipeline` manages the full lifecycle of a model across versions: it monitors
production metrics, decides when to retrain or fine-tune, creates `TrainJob` CRs,
gates deployments on eval results, and manages canary rollout with automatic rollback.

This is what closes the loop between "model is deployed" and "model needs updating."

## How It Works

```
┌─────────────────────────────────────────────────────────────────┐
│  ModelPipeline (always reconciling)                             │
│                                                                 │
│  ┌────────┐   trigger   ┌──────────┐   eval    ┌──────────┐   │
│  │  Idle   │────fires───►│ Training  │──passed──►│ Deploying│   │
│  │         │             │           │           │          │   │
│  │ watches │             │ creates   │           │ annotates│   │
│  │ serving │             │ TrainJob  │           │ TrainJob │   │
│  │ metrics │             │ v(N+1)    │           │ for      │   │
│  │         │             │           │           │ install  │   │
│  └────▲───┘             └─────┬─────┘           └────┬─────┘   │
│       │                       │                       │         │
│       │                       │ failed                │ canary? │
│       │                       ▼                       ▼         │
│       │              stays Idle              ┌──────────┐       │
│       │              (trigger re-fires       │  Canary   │       │
│       │               if conditions hold)    │           │       │
│       │                                      │ monitors  │       │
│       │                                      │ metrics   │       │
│       │                                      └──┬───┬───┘       │
│       │                              regressed  │   │ OK        │
│       │                                         ▼   ▼           │
│       │                              ┌────────┐ ┌────────┐     │
│       │                              │Rollback│ │Promoted│     │
│       │                              └───┬────┘ └───┬────┘     │
│       │                                  │          │           │
│       └──────────────────────────────────┴──────────┘           │
└─────────────────────────────────────────────────────────────────┘
```

## Pipeline Phases

| Phase | What's happening |
|---|---|
| `Idle` | Watching serving metrics, evaluating triggers every 60s |
| `Training` | A TrainJob is running. The pipeline watches its phase. |
| `Evaluating` | The TrainJob is in its eval phase. Pipeline waits. |
| `Deploying` | Eval passed. Pipeline annotates the TrainJob for the model-install agent. |
| `Canary` | New model is serving a fraction of traffic. Pipeline monitors for regression. |
| `Rollback` | Canary regressed. Pipeline reverts to previous serving version. |
| `Paused` | Pipeline is paused (`spec.paused: true`). No new TrainJobs created. |

## Triggers

Triggers define *when* a new training run should start. Multiple triggers are OR'd.

### MetricThreshold

Fires when a metric on the currently serving model crosses a threshold.

```yaml
triggers:
  - name: drift-high
    type: MetricThreshold
    action: Retrain
    metricThreshold:
      metric: serving.embeddingDriftKL
      operator: GreaterThan
      value: 0.15
      cooldownMinutes: 360   # don't retrigger within 6 hours

  - name: accuracy-drop
    type: MetricThreshold
    action: FineTune
    metricThreshold:
      metric: serving.routingAccuracy
      operator: LessThan
      value: 0.85
      cooldownMinutes: 120
```

Supported metrics are dot-paths into `TrainJob.status`:
- `serving.routingAccuracy`, `serving.embeddingDriftKL`, `serving.embeddingDriftMMD`
- `serving.latencyP50Ms`, `serving.latencyP95Ms`, `serving.latencyP99Ms`
- `serving.cacheHitRate`
- `training.mfu`, `training.trainLoss`, `training.commComputeRatio`

### Schedule

Fires on a cron schedule. Implemented via a CronJob that annotates the ModelPipeline.

```yaml
triggers:
  - name: weekly-refresh
    type: Schedule
    action: Retrain
    schedule:
      cron: "0 2 * * 0"   # Sunday 2am
```

### Manual

Fires when a user adds the `training.vsr.dev/trigger` annotation to the ModelPipeline.

```yaml
triggers:
  - name: manual
    type: Manual
    action: Retrain
```

Trigger it: `kubectl annotate modelpipeline clip-vit-l training.vsr.dev/trigger=now`

## Actions: Retrain vs FineTune

When a trigger fires, it specifies an action:

**Retrain**: Creates a fresh TrainJob from the template. Full training from scratch (or from the base model weights). Used when the model has fundamentally drifted or new data changes the distribution significantly.

**FineTune**: Creates a TrainJob that starts from the latest successful checkpoint. The pipeline injects `RESUME_CHECKPOINT` and applies `fineTuneDefaults` (lower LR, fewer steps). Used when the model is mostly good but needs adjustment on specific benchmarks.

```yaml
fineTuneDefaults:
  maxSteps: 5000
  learningRateScale: 0.1
  additionalEnv:
    - name: WARMUP_STEPS
      value: "100"
```

## Versioning

Each training run gets a monotonically increasing version number. The pipeline manages version history and garbage collection.

```yaml
versioning:
  maxConcurrent: 1          # only one training run at a time
  retainVersions: 5         # keep last 5 versions in status.versionHistory
  autoPromote: false        # require eval.verdict == "promote" before deploying
  rollbackOnRegression: true  # rollback canary if metrics degrade
```

The version history in status tracks every run:

```yaml
status:
  currentVersion: 4
  servingVersion: 3
  servingTrainJob: clip-vit-l-v3
  activeTrainJob: clip-vit-l-v4
  versionHistory:
    - version: 2
      trainJobName: clip-vit-l-v2
      action: Retrain
      trigger: weekly-refresh
      phase: Succeeded
      evalVerdict: promote
      promoted: true
      checkpointPath: /checkpoints/v2/step-50000
    - version: 3
      trainJobName: clip-vit-l-v3
      action: FineTune
      trigger: accuracy-drop
      phase: Succeeded
      evalVerdict: promote
      promoted: true
      checkpointPath: /checkpoints/v3/step-5000
    - version: 4
      trainJobName: clip-vit-l-v4
      action: Retrain
      trigger: drift-high
      phase: Running
      promoted: false
```

## Canary Deployment

When `serving.canaryPercent > 0`, the pipeline doesn't do a full rollout immediately.
Instead, it enters the `Canary` phase:

1. The new model serves `canaryPercent` of traffic
2. The pipeline monitors the primary metric (routing accuracy by default)
3. If the canary metric drops below 95% of the baseline, the pipeline rolls back automatically
4. If the canary survives `canaryDurationMinutes`, it gets promoted to full traffic

```yaml
serving:
  vllmDeployment: clip-vit-l-serving
  vllmNamespace: inference
  canaryPercent: 10
  canaryDurationMinutes: 60
```

## Eval Gating

The eval verdict from the TrainJob controls what the pipeline does next:

| Verdict | Pipeline action |
|---|---|
| `promote` | Deploy (or canary) the new model |
| `rollback` | Discard the run, keep the previous model |
| `retrain` | Discard the run, go back to Idle (triggers may re-fire) |
| (no eval configured) | Deploy if `autoPromote: true`, otherwise wait for manual promotion |

## Full Example

```yaml
apiVersion: training.vsr.dev/v1alpha1
kind: ModelPipeline
metadata:
  name: clip-vit-l
spec:
  template:
    metadata:
      labels:
        app: clip-training
    spec:
      model: clip-vit-l
      image: ghcr.io/vsr/clip-training:v1.2
      numNodes: 8
      gpusPerNode: 8
      autoParallelism: true
      nodeSelector:
        nvidia.com/gpu.product: "NVIDIA-H100-SXM5-80GB"
      checkpoint:
        enabled: true
        storagePath: /checkpoints
        validateOnSave: true
      metricsConfig:
        enabled: true
        scrapeIntervalSeconds: 15
      evalConfig:
        datasetPath: /data/eval/mteb-retrieval
        gpusPerNode: 1
      maxRuntime: "48h"

  triggers:
    - name: drift-high
      type: MetricThreshold
      action: Retrain
      metricThreshold:
        metric: serving.embeddingDriftKL
        operator: GreaterThan
        value: 0.15
        cooldownMinutes: 360

    - name: accuracy-drop
      type: MetricThreshold
      action: FineTune
      metricThreshold:
        metric: serving.routingAccuracy
        operator: LessThan
        value: 0.85
        cooldownMinutes: 120

    - name: weekly-refresh
      type: Schedule
      action: Retrain
      schedule:
        cron: "0 2 * * 0"

    - name: manual
      type: Manual

  fineTuneDefaults:
    maxSteps: 5000
    learningRateScale: 0.1

  versioning:
    maxConcurrent: 1
    retainVersions: 5
    autoPromote: true
    rollbackOnRegression: true

  serving:
    vllmDeployment: clip-vit-l-serving
    vllmNamespace: inference
    semanticRouterConfigMap: semantic-router-config
    canaryPercent: 10
    canaryDurationMinutes: 60
```

This pipeline:
1. Watches the currently serving model's routing accuracy and embedding drift
2. Fine-tunes when accuracy drops below 0.85 (from latest checkpoint, 5000 steps, 0.1x LR)
3. Retrains from scratch when drift exceeds 0.15
4. Also retrains weekly on a schedule
5. After training, runs eval. Only deploys if verdict is "promote"
6. Deploys as a 10% canary for 60 minutes
7. Auto-rolls back if canary metrics regress by more than 5%
8. Keeps the last 5 versions of history

## How Agents Interact with ModelPipeline

The agents don't create ModelPipelines or TrainJobs directly. Instead:

- **model-trainer agent** improves the template spec and fine-tune defaults in the ModelPipeline
- **infra agent** tunes the auto-parallelism config and node selectors in the template
- **sre agent** adjusts trigger thresholds and canary durations based on observed patterns
- **model-eval agent** improves eval benchmarks and regression thresholds
- **model-install agent** consumes the `training.vsr.dev/promote` annotation on TrainJobs to generate deployment artifacts
- **ops agent** monitors pipeline health and adjusts versioning policies
