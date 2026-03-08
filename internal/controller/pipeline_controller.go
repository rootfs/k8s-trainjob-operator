package controller

import (
	"context"
	"fmt"
	"strconv"
	"strings"
	"time"

	"k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/tools/record"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/log"

	aiv1 "github.com/rootfs/trainjob-operator/api/v1alpha1"
)

const (
	pipelinePollInterval   = 60 * time.Second
	defaultRetainVersions  = 5
	defaultCooldownMinutes = 120
)

// ModelPipelineReconciler manages the continuous lifecycle of a model.
// It watches production metrics, evaluates triggers, creates TrainJobs,
// gates deployments on eval results, and manages model versioning.
type ModelPipelineReconciler struct {
	client.Client
	Recorder record.EventRecorder
}

//+kubebuilder:rbac:groups=training.vsr.dev,resources=modelpipelines,verbs=get;list;watch;create;update;patch;delete
//+kubebuilder:rbac:groups=training.vsr.dev,resources=modelpipelines/status,verbs=get;update;patch
//+kubebuilder:rbac:groups=training.vsr.dev,resources=trainjobs,verbs=get;list;watch;create;update;patch;delete

func (r *ModelPipelineReconciler) Reconcile(ctx context.Context, req ctrl.Request) (ctrl.Result, error) {
	logger := log.FromContext(ctx)

	var pipeline aiv1.ModelPipeline
	if err := r.Get(ctx, req.NamespacedName, &pipeline); err != nil {
		if errors.IsNotFound(err) {
			return ctrl.Result{}, nil
		}
		return ctrl.Result{}, err
	}

	// Handle paused pipelines
	if pipeline.Spec.Paused {
		if pipeline.Status.Phase != aiv1.PipelinePaused {
			pipeline.Status.Phase = aiv1.PipelinePaused
			meta.SetStatusCondition(&pipeline.Status.Conditions, metav1.Condition{
				Type:               "Paused",
				Status:             metav1.ConditionTrue,
				Reason:             "UserPaused",
				Message:            "Pipeline paused by user",
				LastTransitionTime: metav1.Now(),
			})
			if err := r.Status().Update(ctx, &pipeline); err != nil {
				return ctrl.Result{}, err
			}
		}
		return ctrl.Result{}, nil
	}

	logger.Info("Reconciling pipeline", "phase", pipeline.Status.Phase, "name", pipeline.Name)

	switch pipeline.Status.Phase {
	case "", aiv1.PipelineIdle:
		return r.handleIdle(ctx, &pipeline)
	case aiv1.PipelineTraining:
		return r.handleTraining(ctx, &pipeline)
	case aiv1.PipelineEvaluating:
		return r.handleEvaluating(ctx, &pipeline)
	case aiv1.PipelineDeploying:
		return r.handleDeploying(ctx, &pipeline)
	case aiv1.PipelineCanary:
		return r.handleCanary(ctx, &pipeline)
	case aiv1.PipelineRollback:
		return r.handleRollback(ctx, &pipeline)
	case aiv1.PipelinePaused:
		// Unpaused — transition back to idle
		return r.pipelineTransition(ctx, &pipeline, aiv1.PipelineIdle, "Unpaused", "Pipeline resumed")
	default:
		logger.Error(nil, "Unknown pipeline phase", "phase", pipeline.Status.Phase)
		return ctrl.Result{}, nil
	}
}

// ════════════════════════════════════════════════════════════════
//  Phase handlers
// ════════════════════════════════════════════════════════════════

// handleIdle checks triggers and creates a new TrainJob if any fires.
func (r *ModelPipelineReconciler) handleIdle(ctx context.Context, pipeline *aiv1.ModelPipeline) (ctrl.Result, error) {
	logger := log.FromContext(ctx)

	// Check each trigger
	firedTrigger, action := r.evaluateTriggers(ctx, pipeline)
	if firedTrigger == "" {
		return ctrl.Result{RequeueAfter: pipelinePollInterval}, nil
	}

	logger.Info("Trigger fired", "trigger", firedTrigger, "action", action)

	// Check max concurrent
	maxConcurrent := int32(1)
	if pipeline.Spec.Versioning.MaxConcurrent != nil {
		maxConcurrent = *pipeline.Spec.Versioning.MaxConcurrent
	}
	if maxConcurrent > 0 && pipeline.Status.ActiveTrainJob != "" {
		logger.Info("TrainJob already active, skipping trigger", "active", pipeline.Status.ActiveTrainJob)
		return ctrl.Result{RequeueAfter: pipelinePollInterval}, nil
	}

	// Bump version
	pipeline.Status.CurrentVersion++
	version := pipeline.Status.CurrentVersion

	// Create TrainJob from template
	tj := r.buildTrainJobFromTemplate(pipeline, version, firedTrigger, action)
	if err := r.Create(ctx, tj); err != nil {
		if errors.IsAlreadyExists(err) {
			logger.Info("TrainJob already exists", "name", tj.Name)
		} else {
			return ctrl.Result{}, fmt.Errorf("creating TrainJob v%d: %w", version, err)
		}
	}

	now := metav1.Now()
	pipeline.Status.ActiveTrainJob = tj.Name
	pipeline.Status.ActiveVersion = version
	pipeline.Status.LastTrigger = firedTrigger
	pipeline.Status.LastTriggerTime = &now

	r.Recorder.Event(pipeline, "Normal", "TrainJobCreated",
		fmt.Sprintf("Created TrainJob %s (v%d) triggered by %s, action=%s",
			tj.Name, version, firedTrigger, action))

	return r.pipelineTransition(ctx, pipeline, aiv1.PipelineTraining, "TrainJobCreated",
		fmt.Sprintf("Training v%d started", version))
}

// handleTraining watches the active TrainJob and transitions when it completes.
func (r *ModelPipelineReconciler) handleTraining(ctx context.Context, pipeline *aiv1.ModelPipeline) (ctrl.Result, error) {
	if pipeline.Status.ActiveTrainJob == "" {
		return r.pipelineTransition(ctx, pipeline, aiv1.PipelineIdle, "NoActiveJob", "No active TrainJob")
	}

	var tj aiv1.TrainJob
	key := client.ObjectKey{Namespace: pipeline.Namespace, Name: pipeline.Status.ActiveTrainJob}
	if err := r.Get(ctx, key, &tj); err != nil {
		if errors.IsNotFound(err) {
			return r.pipelineTransition(ctx, pipeline, aiv1.PipelineIdle, "TrainJobLost",
				"Active TrainJob disappeared")
		}
		return ctrl.Result{}, err
	}

	switch tj.Status.Phase {
	case aiv1.PhaseSucceeded:
		// Training (and possibly eval) completed — check eval results
		if tj.Status.Eval != nil {
			return r.handleEvalResult(ctx, pipeline, &tj)
		}
		// No eval configured — go straight to deploying if autoPromote
		if pipeline.Spec.Versioning.AutoPromote {
			return r.pipelineTransition(ctx, pipeline, aiv1.PipelineDeploying, "TrainingSucceeded",
				"Training completed, auto-promoting")
		}
		r.recordVersion(pipeline, &tj)
		pipeline.Status.ActiveTrainJob = ""
		return r.pipelineTransition(ctx, pipeline, aiv1.PipelineIdle, "TrainingSucceeded",
			"Training completed, awaiting manual promotion")

	case aiv1.PhaseFailed:
		r.recordVersion(pipeline, &tj)
		pipeline.Status.ActiveTrainJob = ""
		r.Recorder.Event(pipeline, "Warning", "TrainingFailed",
			fmt.Sprintf("TrainJob %s failed: %s", tj.Name, tj.Status.FailureMessage))
		return r.pipelineTransition(ctx, pipeline, aiv1.PipelineIdle, "TrainingFailed",
			"Training failed: "+tj.Status.FailureMessage)

	case aiv1.PhaseEvaluating:
		// TrainJob is in its eval phase — wait
		return ctrl.Result{RequeueAfter: pipelinePollInterval}, nil

	default:
		// Still training — requeue
		return ctrl.Result{RequeueAfter: pipelinePollInterval}, nil
	}
}

// handleEvalResult processes the eval verdict and decides next steps.
func (r *ModelPipelineReconciler) handleEvalResult(ctx context.Context, pipeline *aiv1.ModelPipeline, tj *aiv1.TrainJob) (ctrl.Result, error) {
	verdict := tj.Status.Eval.Verdict

	switch verdict {
	case "promote":
		r.Recorder.Event(pipeline, "Normal", "EvalPassed",
			fmt.Sprintf("v%d eval verdict: promote", pipeline.Status.ActiveVersion))
		if pipeline.Spec.Versioning.AutoPromote {
			return r.pipelineTransition(ctx, pipeline, aiv1.PipelineDeploying, "EvalPromote",
				fmt.Sprintf("v%d passed eval, deploying", pipeline.Status.ActiveVersion))
		}
		r.recordVersion(pipeline, tj)
		pipeline.Status.ActiveTrainJob = ""
		return r.pipelineTransition(ctx, pipeline, aiv1.PipelineIdle, "EvalPromote",
			"Eval passed, awaiting manual deployment")

	case "rollback":
		r.Recorder.Event(pipeline, "Warning", "EvalRollback",
			fmt.Sprintf("v%d eval verdict: rollback — worse than previous", pipeline.Status.ActiveVersion))
		r.recordVersion(pipeline, tj)
		pipeline.Status.ActiveTrainJob = ""
		return r.pipelineTransition(ctx, pipeline, aiv1.PipelineIdle, "EvalRollback",
			"Eval recommends rollback — keeping previous model")

	case "retrain":
		r.Recorder.Event(pipeline, "Warning", "EvalRetrain",
			fmt.Sprintf("v%d eval verdict: retrain — needs improvement", pipeline.Status.ActiveVersion))
		r.recordVersion(pipeline, tj)
		pipeline.Status.ActiveTrainJob = ""
		// Go back to idle — the trigger system will re-fire if conditions are still met
		return r.pipelineTransition(ctx, pipeline, aiv1.PipelineIdle, "EvalRetrain",
			"Eval recommends retraining with adjusted config")

	default:
		// No verdict or unknown — treat as promote if autoPromote
		if pipeline.Spec.Versioning.AutoPromote {
			return r.pipelineTransition(ctx, pipeline, aiv1.PipelineDeploying, "EvalNoVerdict",
				"No eval verdict, auto-promoting")
		}
		r.recordVersion(pipeline, tj)
		pipeline.Status.ActiveTrainJob = ""
		return r.pipelineTransition(ctx, pipeline, aiv1.PipelineIdle, "EvalNoVerdict",
			"Eval completed without verdict, awaiting manual decision")
	}
}

// handleEvaluating is a pass-through. Eval is now handled within the TrainJob reconciler.
// The pipeline watches the TrainJob phase in handleTraining.
func (r *ModelPipelineReconciler) handleEvaluating(ctx context.Context, pipeline *aiv1.ModelPipeline) (ctrl.Result, error) {
	return r.handleTraining(ctx, pipeline)
}

// handleDeploying manages the model-install step. In practice, this annotates
// the TrainJob to signal the model-install agent or creates a conversion Job.
// For now, it transitions to Canary if canary is configured, or to Idle.
func (r *ModelPipelineReconciler) handleDeploying(ctx context.Context, pipeline *aiv1.ModelPipeline) (ctrl.Result, error) {
	if pipeline.Status.ActiveTrainJob == "" {
		return r.pipelineTransition(ctx, pipeline, aiv1.PipelineIdle, "NoActiveJob", "No active TrainJob to deploy")
	}

	// Annotate the TrainJob to signal the model-install pipeline
	var tj aiv1.TrainJob
	key := client.ObjectKey{Namespace: pipeline.Namespace, Name: pipeline.Status.ActiveTrainJob}
	if err := r.Get(ctx, key, &tj); err != nil {
		return ctrl.Result{}, err
	}

	if tj.Annotations == nil {
		tj.Annotations = make(map[string]string)
	}
	tj.Annotations["training.vsr.dev/promote"] = "true"
	tj.Annotations["training.vsr.dev/pipeline"] = pipeline.Name
	tj.Annotations["training.vsr.dev/version"] = strconv.Itoa(int(pipeline.Status.ActiveVersion))
	if err := r.Update(ctx, &tj); err != nil {
		return ctrl.Result{}, err
	}

	r.Recorder.Event(pipeline, "Normal", "DeploymentInitiated",
		fmt.Sprintf("Annotated TrainJob %s for deployment", tj.Name))

	// If canary is configured, transition to canary
	if pipeline.Spec.Serving != nil && pipeline.Spec.Serving.CanaryPercent != nil && *pipeline.Spec.Serving.CanaryPercent > 0 {
		now := metav1.Now()
		pipeline.Status.CanaryStatus = &aiv1.CanaryStatus{
			Version:        pipeline.Status.ActiveVersion,
			StartedAt:      &now,
			TrafficPercent: *pipeline.Spec.Serving.CanaryPercent,
		}
		return r.pipelineTransition(ctx, pipeline, aiv1.PipelineCanary, "CanaryStarted",
			fmt.Sprintf("Canary deployment started at %d%% traffic", *pipeline.Spec.Serving.CanaryPercent))
	}

	// No canary — full rollout, record and go to idle
	r.recordVersion(pipeline, &tj)
	r.promoteVersion(pipeline)
	pipeline.Status.ActiveTrainJob = ""
	return r.pipelineTransition(ctx, pipeline, aiv1.PipelineIdle, "Deployed",
		fmt.Sprintf("v%d deployed to production (full rollout)", pipeline.Status.ActiveVersion))
}

// handleCanary monitors the canary deployment and decides to promote or rollback.
func (r *ModelPipelineReconciler) handleCanary(ctx context.Context, pipeline *aiv1.ModelPipeline) (ctrl.Result, error) {
	canary := pipeline.Status.CanaryStatus
	if canary == nil || canary.StartedAt == nil {
		return r.pipelineTransition(ctx, pipeline, aiv1.PipelineIdle, "NoCanary", "No canary status")
	}

	// Check canary duration
	canaryDuration := 60 * time.Minute
	if pipeline.Spec.Serving != nil && pipeline.Spec.Serving.CanaryDurationMinutes != nil {
		canaryDuration = time.Duration(*pipeline.Spec.Serving.CanaryDurationMinutes) * time.Minute
	}

	elapsed := time.Since(canary.StartedAt.Time)

	// Read serving metrics from the active TrainJob to compare canary vs baseline
	if pipeline.Status.ActiveTrainJob != "" {
		var tj aiv1.TrainJob
		key := client.ObjectKey{Namespace: pipeline.Namespace, Name: pipeline.Status.ActiveTrainJob}
		if err := r.Get(ctx, key, &tj); err == nil && tj.Status.Serving != nil {
			if tj.Status.Serving.RoutingAccuracy != nil {
				canary.CanaryMetricValue = tj.Status.Serving.RoutingAccuracy
			}
		}
	}

	// Check for regression during canary
	rollbackOnRegression := true
	if pipeline.Spec.Versioning.RollbackOnRegression != nil {
		rollbackOnRegression = *pipeline.Spec.Versioning.RollbackOnRegression
	}

	if rollbackOnRegression && canary.BaselineMetricValue != nil && canary.CanaryMetricValue != nil {
		if *canary.CanaryMetricValue < *canary.BaselineMetricValue*0.95 {
			r.Recorder.Event(pipeline, "Warning", "CanaryRegression",
				fmt.Sprintf("Canary v%d metric %.3f < baseline %.3f, rolling back",
					canary.Version, *canary.CanaryMetricValue, *canary.BaselineMetricValue))
			pipeline.Status.CanaryStatus = nil
			return r.pipelineTransition(ctx, pipeline, aiv1.PipelineRollback, "CanaryRegression",
				"Canary metrics regressed, rolling back")
		}
	}

	if elapsed >= canaryDuration {
		// Canary period complete — promote
		r.Recorder.Event(pipeline, "Normal", "CanaryPromoted",
			fmt.Sprintf("Canary v%d promoted after %s", canary.Version, canaryDuration))

		if pipeline.Status.ActiveTrainJob != "" {
			var tj aiv1.TrainJob
			key := client.ObjectKey{Namespace: pipeline.Namespace, Name: pipeline.Status.ActiveTrainJob}
			if err := r.Get(ctx, key, &tj); err == nil {
				r.recordVersion(pipeline, &tj)
			}
		}
		r.promoteVersion(pipeline)
		pipeline.Status.CanaryStatus = nil
		pipeline.Status.ActiveTrainJob = ""
		return r.pipelineTransition(ctx, pipeline, aiv1.PipelineIdle, "CanaryPromoted",
			fmt.Sprintf("v%d promoted to production", canary.Version))
	}

	// Still in canary window
	if err := r.Status().Update(ctx, pipeline); err != nil {
		return ctrl.Result{}, err
	}
	return ctrl.Result{RequeueAfter: pipelinePollInterval}, nil
}

// handleRollback reverts to the previous serving version.
func (r *ModelPipelineReconciler) handleRollback(ctx context.Context, pipeline *aiv1.ModelPipeline) (ctrl.Result, error) {
	r.Recorder.Event(pipeline, "Warning", "RolledBack",
		fmt.Sprintf("Rolled back to v%d", pipeline.Status.ServingVersion))

	if pipeline.Status.ActiveTrainJob != "" {
		var tj aiv1.TrainJob
		key := client.ObjectKey{Namespace: pipeline.Namespace, Name: pipeline.Status.ActiveTrainJob}
		if err := r.Get(ctx, key, &tj); err == nil {
			r.recordVersion(pipeline, &tj)
		}
	}

	pipeline.Status.ActiveTrainJob = ""
	pipeline.Status.CanaryStatus = nil
	return r.pipelineTransition(ctx, pipeline, aiv1.PipelineIdle, "RollbackComplete",
		fmt.Sprintf("Rolled back to v%d", pipeline.Status.ServingVersion))
}

// ════════════════════════════════════════════════════════════════
//  Trigger evaluation
// ════════════════════════════════════════════════════════════════

// evaluateTriggers checks all configured triggers and returns the first one that fires.
func (r *ModelPipelineReconciler) evaluateTriggers(ctx context.Context, pipeline *aiv1.ModelPipeline) (string, aiv1.PipelineAction) {
	for _, trigger := range pipeline.Spec.Triggers {
		action := trigger.Action
		if action == "" {
			action = aiv1.ActionRetrain
		}

		switch trigger.Type {
		case aiv1.TriggerMetricThreshold:
			if trigger.MetricThreshold == nil {
				continue
			}
			if r.checkCooldown(pipeline, trigger.Name, trigger.MetricThreshold.CooldownMinutes) {
				continue
			}
			if r.evaluateMetricThreshold(ctx, pipeline, trigger.MetricThreshold) {
				return trigger.Name, action
			}

		case aiv1.TriggerManual:
			if r.checkManualTrigger(pipeline) {
				return trigger.Name, action
			}

		case aiv1.TriggerSchedule:
			// Schedule triggers would be handled by a CronJob creating an annotation.
			// The controller checks for the annotation here.
			if r.checkScheduleTrigger(pipeline, trigger.Name) {
				return trigger.Name, action
			}
		}
	}
	return "", ""
}

// evaluateMetricThreshold reads the serving TrainJob's status and checks the metric.
func (r *ModelPipelineReconciler) evaluateMetricThreshold(ctx context.Context, pipeline *aiv1.ModelPipeline, mt *aiv1.MetricThresholdTrigger) bool {
	if pipeline.Status.ServingTrainJob == "" {
		return false
	}

	var tj aiv1.TrainJob
	key := client.ObjectKey{Namespace: pipeline.Namespace, Name: pipeline.Status.ServingTrainJob}
	if err := r.Get(ctx, key, &tj); err != nil {
		return false
	}

	metricValue, found := resolveMetric(&tj, mt.Metric)
	if !found {
		return false
	}

	switch mt.Operator {
	case aiv1.OperatorLessThan:
		return metricValue < mt.Value
	case aiv1.OperatorGreaterThan:
		return metricValue > mt.Value
	}
	return false
}

// resolveMetric reads a dot-path metric from TrainJob status.
func resolveMetric(tj *aiv1.TrainJob, path string) (float64, bool) {
	parts := strings.SplitN(path, ".", 2)
	if len(parts) < 2 {
		return 0, false
	}

	switch parts[0] {
	case "serving":
		if tj.Status.Serving == nil {
			return 0, false
		}
		return resolveServingMetric(tj.Status.Serving, parts[1])
	case "training":
		if tj.Status.Training == nil {
			return 0, false
		}
		return resolveTrainingMetric(tj.Status.Training, parts[1])
	}
	return 0, false
}

func resolveServingMetric(s *aiv1.ServingMetrics, field string) (float64, bool) {
	switch field {
	case "routingAccuracy":
		if s.RoutingAccuracy != nil {
			return *s.RoutingAccuracy, true
		}
	case "routingAccuracyP7d":
		if s.RoutingAccuracyP7d != nil {
			return *s.RoutingAccuracyP7d, true
		}
	case "embeddingDriftKL":
		if s.EmbeddingDriftKL != nil {
			return *s.EmbeddingDriftKL, true
		}
	case "embeddingDriftMMD":
		if s.EmbeddingDriftMMD != nil {
			return *s.EmbeddingDriftMMD, true
		}
	case "latencyP50Ms":
		if s.LatencyP50Ms != nil {
			return *s.LatencyP50Ms, true
		}
	case "latencyP95Ms":
		if s.LatencyP95Ms != nil {
			return *s.LatencyP95Ms, true
		}
	case "latencyP99Ms":
		if s.LatencyP99Ms != nil {
			return *s.LatencyP99Ms, true
		}
	case "cacheHitRate":
		if s.CacheHitRate != nil {
			return *s.CacheHitRate, true
		}
	}
	return 0, false
}

func resolveTrainingMetric(t *aiv1.TrainingMetrics, field string) (float64, bool) {
	switch field {
	case "mfu":
		if t.MFU != nil {
			return *t.MFU, true
		}
	case "trainLoss":
		if t.TrainLoss != nil {
			return *t.TrainLoss, true
		}
	case "commComputeRatio":
		if t.CommComputeRatio != nil {
			return *t.CommComputeRatio, true
		}
	}
	return 0, false
}

func (r *ModelPipelineReconciler) checkCooldown(pipeline *aiv1.ModelPipeline, triggerName string, cooldownMinutes *int32) bool {
	if pipeline.Status.LastTriggerTime == nil || pipeline.Status.LastTrigger != triggerName {
		return false
	}
	cooldown := int32(defaultCooldownMinutes)
	if cooldownMinutes != nil {
		cooldown = *cooldownMinutes
	}
	return time.Since(pipeline.Status.LastTriggerTime.Time) < time.Duration(cooldown)*time.Minute
}

func (r *ModelPipelineReconciler) checkManualTrigger(pipeline *aiv1.ModelPipeline) bool {
	if pipeline.Annotations == nil {
		return false
	}
	_, ok := pipeline.Annotations["training.vsr.dev/trigger"]
	return ok
}

func (r *ModelPipelineReconciler) checkScheduleTrigger(pipeline *aiv1.ModelPipeline, triggerName string) bool {
	if pipeline.Annotations == nil {
		return false
	}
	v, ok := pipeline.Annotations["training.vsr.dev/schedule-trigger"]
	return ok && v == triggerName
}

// ════════════════════════════════════════════════════════════════
//  TrainJob creation
// ════════════════════════════════════════════════════════════════

func (r *ModelPipelineReconciler) buildTrainJobFromTemplate(
	pipeline *aiv1.ModelPipeline,
	version int32,
	triggerName string,
	action aiv1.PipelineAction,
) *aiv1.TrainJob {
	spec := pipeline.Spec.Template.Spec.DeepCopy()

	// For fine-tune: start from the latest checkpoint and apply overrides
	if action == aiv1.ActionFineTune {
		if latest := r.latestCheckpointPath(pipeline); latest != "" {
			spec.Env = append(spec.Env, aiv1.EnvVar{
				Name:  "RESUME_CHECKPOINT",
				Value: latest,
			})
		}

		if ft := pipeline.Spec.FineTuneDefaults; ft != nil {
			if ft.MaxSteps != nil {
				spec.Env = append(spec.Env, aiv1.EnvVar{
					Name:  "MAX_STEPS",
					Value: strconv.FormatInt(*ft.MaxSteps, 10),
				})
			}
			if ft.LearningRateScale != nil {
				spec.Env = append(spec.Env, aiv1.EnvVar{
					Name:  "LR_SCALE",
					Value: strconv.FormatFloat(*ft.LearningRateScale, 'f', -1, 64),
				})
			}
			for _, e := range ft.AdditionalEnv {
				spec.Env = append(spec.Env, e)
			}
		}
	}

	// Version-specific checkpoint path
	spec.Checkpoint.StoragePath = fmt.Sprintf("%s/v%d", spec.Checkpoint.StoragePath, version)

	labels := make(map[string]string)
	for k, v := range pipeline.Spec.Template.Labels {
		labels[k] = v
	}
	labels["training.vsr.dev/pipeline"] = pipeline.Name
	labels["training.vsr.dev/version"] = strconv.Itoa(int(version))
	labels["training.vsr.dev/action"] = string(action)

	annotations := make(map[string]string)
	for k, v := range pipeline.Spec.Template.Annotations {
		annotations[k] = v
	}
	annotations["training.vsr.dev/trigger"] = triggerName

	return &aiv1.TrainJob{
		ObjectMeta: metav1.ObjectMeta{
			Name:        fmt.Sprintf("%s-v%d", pipeline.Name, version),
			Namespace:   pipeline.Namespace,
			Labels:      labels,
			Annotations: annotations,
			OwnerReferences: []metav1.OwnerReference{
				{
					APIVersion: aiv1.GroupVersion.String(),
					Kind:       "ModelPipeline",
					Name:       pipeline.Name,
					UID:        pipeline.UID,
				},
			},
		},
		Spec: *spec,
	}
}

func (r *ModelPipelineReconciler) latestCheckpointPath(pipeline *aiv1.ModelPipeline) string {
	for i := len(pipeline.Status.VersionHistory) - 1; i >= 0; i-- {
		v := pipeline.Status.VersionHistory[i]
		if v.CheckpointPath != "" && v.Phase == aiv1.PhaseSucceeded {
			return v.CheckpointPath
		}
	}
	return ""
}

// ════════════════════════════════════════════════════════════════
//  Version management
// ════════════════════════════════════════════════════════════════

func (r *ModelPipelineReconciler) recordVersion(pipeline *aiv1.ModelPipeline, tj *aiv1.TrainJob) {
	action := aiv1.PipelineAction(tj.Labels["training.vsr.dev/action"])
	trigger := tj.Annotations["training.vsr.dev/trigger"]

	mv := aiv1.ModelVersion{
		Version:      pipeline.Status.ActiveVersion,
		TrainJobName: tj.Name,
		Action:       action,
		Trigger:      trigger,
		Phase:        tj.Status.Phase,
		StartedAt:    tj.Status.StartTime,
		CompletedAt:  tj.Status.CompletionTime,
	}

	if tj.Status.Eval != nil {
		mv.EvalVerdict = tj.Status.Eval.Verdict
	}
	if tj.Status.LastCheckpoint != "" {
		mv.CheckpointPath = tj.Status.LastCheckpoint
	}

	pipeline.Status.VersionHistory = append(pipeline.Status.VersionHistory, mv)

	// Garbage collect old versions
	retain := int32(defaultRetainVersions)
	if pipeline.Spec.Versioning.RetainVersions != nil {
		retain = *pipeline.Spec.Versioning.RetainVersions
	}
	if int32(len(pipeline.Status.VersionHistory)) > retain {
		pipeline.Status.VersionHistory = pipeline.Status.VersionHistory[len(pipeline.Status.VersionHistory)-int(retain):]
	}
}

func (r *ModelPipelineReconciler) promoteVersion(pipeline *aiv1.ModelPipeline) {
	pipeline.Status.ServingVersion = pipeline.Status.ActiveVersion
	pipeline.Status.ServingTrainJob = pipeline.Status.ActiveTrainJob

	// Mark the version as promoted in history
	for i := len(pipeline.Status.VersionHistory) - 1; i >= 0; i-- {
		if pipeline.Status.VersionHistory[i].Version == pipeline.Status.ActiveVersion {
			pipeline.Status.VersionHistory[i].Promoted = true
			break
		}
	}
}

// ════════════════════════════════════════════════════════════════
//  Helpers
// ════════════════════════════════════════════════════════════════

func (r *ModelPipelineReconciler) pipelineTransition(
	ctx context.Context,
	pipeline *aiv1.ModelPipeline,
	phase aiv1.PipelinePhase,
	reason, message string,
) (ctrl.Result, error) {
	logger := log.FromContext(ctx)
	logger.Info("Pipeline phase transition", "from", pipeline.Status.Phase, "to", phase, "reason", reason)

	pipeline.Status.Phase = phase
	meta.SetStatusCondition(&pipeline.Status.Conditions, metav1.Condition{
		Type:               string(phase),
		Status:             metav1.ConditionTrue,
		Reason:             reason,
		Message:            message,
		LastTransitionTime: metav1.Now(),
	})

	if err := r.Status().Update(ctx, pipeline); err != nil {
		return ctrl.Result{}, fmt.Errorf("updating pipeline status to %s: %w", phase, err)
	}

	if phase == aiv1.PipelineIdle {
		return ctrl.Result{RequeueAfter: pipelinePollInterval}, nil
	}
	return ctrl.Result{Requeue: true}, nil
}

// ════════════════════════════════════════════════════════════════
//  Controller setup
// ════════════════════════════════════════════════════════════════

func (r *ModelPipelineReconciler) SetupWithManager(mgr ctrl.Manager) error {
	return ctrl.NewControllerManagedBy(mgr).
		For(&aiv1.ModelPipeline{}).
		Owns(&aiv1.TrainJob{}).
		Complete(r)
}
