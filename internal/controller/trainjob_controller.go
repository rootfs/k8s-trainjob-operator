package controller

import (
	"context"
	"fmt"
	"time"

	appsv1 "k8s.io/api/apps/v1"
	batchv1 "k8s.io/api/batch/v1"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/tools/record"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/controller/controllerutil"
	"sigs.k8s.io/controller-runtime/pkg/log"

	aiv1 "github.com/rootfs/trainjob-operator/api/v1alpha1"
)

const (
	finalizerName = "training.vsr.dev/checkpoint-protection"

	// Requeue intervals
	prologPollInterval     = 15 * time.Second
	runningPollInterval    = 60 * time.Second
	checkpointPollInterval = 10 * time.Second
)

// TrainJobReconciler reconciles TrainJob objects through a state machine.
//
// State machine:
//
//	Pending → PrologRunning → PrologPassed → Running → Succeeded
//	                ↓                           ↓
//	          PrologFailed                    Failed
//	                                            ↓
//	                                      Checkpointing → Running (retry)
type TrainJobReconciler struct {
	client.Client
	Recorder record.EventRecorder
}

//+kubebuilder:rbac:groups=training.vsr.dev,resources=trainjobs,verbs=get;list;watch;create;update;patch;delete
//+kubebuilder:rbac:groups=training.vsr.dev,resources=trainjobs/status,verbs=get;update;patch
//+kubebuilder:rbac:groups=training.vsr.dev,resources=trainjobs/finalizers,verbs=update
//+kubebuilder:rbac:groups=batch,resources=jobs,verbs=get;list;watch;create;update;patch;delete
//+kubebuilder:rbac:groups=apps,resources=statefulsets,verbs=get;list;watch;create;update;patch;delete
//+kubebuilder:rbac:groups="",resources=services,verbs=get;list;watch;create;update;patch;delete
//+kubebuilder:rbac:groups="",resources=events,verbs=create;patch

func (r *TrainJobReconciler) Reconcile(ctx context.Context, req ctrl.Request) (ctrl.Result, error) {
	logger := log.FromContext(ctx)

	// ── Step 1: FETCH ──
	var tj aiv1.TrainJob
	if err := r.Get(ctx, req.NamespacedName, &tj); err != nil {
		if errors.IsNotFound(err) {
			return ctrl.Result{}, nil
		}
		return ctrl.Result{}, err
	}

	// ── Step 2: FINALIZER ──
	if tj.DeletionTimestamp != nil {
		return r.handleDeletion(ctx, &tj)
	}
	if !controllerutil.ContainsFinalizer(&tj, finalizerName) {
		controllerutil.AddFinalizer(&tj, finalizerName)
		if err := r.Update(ctx, &tj); err != nil {
			return ctrl.Result{}, err
		}
	}

	// ── Step 3: SUSPEND CHECK ──
	// If spec.suspend is true, don't create any child resources.
	// This is the integration point for Kueue: Kueue sets suspend=true
	// until quota is available, then sets suspend=false to release the job.
	if tj.Spec.Suspend != nil && *tj.Spec.Suspend {
		if tj.Status.Phase != aiv1.PhaseSuspended {
			logger.Info("Job suspended by external controller")
			r.Recorder.Event(&tj, corev1.EventTypeNormal, "Suspended",
				"Job suspended, waiting for admission (e.g., Kueue quota)")
			return r.transition(ctx, &tj, aiv1.PhaseSuspended, "Suspended",
				"Waiting for external admission controller to unsuspend")
		}
		return ctrl.Result{}, nil
	}

	// If we were suspended and are now unsuspended, transition to Pending
	if tj.Status.Phase == aiv1.PhaseSuspended {
		logger.Info("Job unsuspended, starting")
		r.Recorder.Event(&tj, corev1.EventTypeNormal, "Unsuspended",
			"Job admitted, proceeding to pending")
		return r.transition(ctx, &tj, aiv1.PhasePending, "Admitted",
			"External admission controller released the job")
	}

	// ── Step 4: STATE MACHINE ──
	logger.Info("Reconciling", "phase", tj.Status.Phase, "name", tj.Name)

	switch tj.Status.Phase {
	case "", aiv1.PhasePending:
		return r.handlePending(ctx, &tj)
	case aiv1.PhasePrologRunning:
		return r.handlePrologRunning(ctx, &tj)
	case aiv1.PhasePrologPassed:
		return r.handlePrologPassed(ctx, &tj)
	case aiv1.PhasePrologFailed:
		return r.handlePrologFailed(ctx, &tj)
	case aiv1.PhaseRunning:
		return r.handleRunning(ctx, &tj)
	case aiv1.PhaseCheckpointing:
		return r.handleCheckpointing(ctx, &tj)
	case aiv1.PhaseEvaluating:
		return r.handleEvaluating(ctx, &tj)
	case aiv1.PhaseSucceeded, aiv1.PhaseFailed:
		return ctrl.Result{}, nil // Terminal states
	default:
		logger.Error(nil, "Unknown phase", "phase", tj.Status.Phase)
		return ctrl.Result{}, nil
	}
}

// ════════════════════════════════════════════════════════════════
//  Phase handlers
// ════════════════════════════════════════════════════════════════

// handlePending launches the prolog health check (or skips it).
func (r *TrainJobReconciler) handlePending(ctx context.Context, tj *aiv1.TrainJob) (ctrl.Result, error) {
	if tj.Spec.SkipProlog {
		return r.transition(ctx, tj, aiv1.PhasePrologPassed, "PrologSkipped", "Prolog skipped per spec")
	}

	prologJob := buildPrologJob(tj)
	if err := controllerutil.SetControllerReference(tj, prologJob, r.Scheme()); err != nil {
		return ctrl.Result{}, fmt.Errorf("setting owner ref on prolog job: %w", err)
	}

	if err := r.Create(ctx, prologJob); err != nil {
		if errors.IsAlreadyExists(err) {
			// Already created on a previous reconcile; move to next phase
			return r.transition(ctx, tj, aiv1.PhasePrologRunning, "PrologCreated", "Prolog job already exists")
		}
		return ctrl.Result{}, fmt.Errorf("creating prolog job: %w", err)
	}

	r.Recorder.Event(tj, corev1.EventTypeNormal, "PrologStarted",
		fmt.Sprintf("Started GPU health check on %d nodes", tj.Spec.NumNodes))

	return r.transition(ctx, tj, aiv1.PhasePrologRunning, "PrologCreated", "Prolog health check job created")
}

// handlePrologRunning polls the prolog Job until it completes.
func (r *TrainJobReconciler) handlePrologRunning(ctx context.Context, tj *aiv1.TrainJob) (ctrl.Result, error) {
	var prologJob batchv1.Job
	key := client.ObjectKey{Namespace: tj.Namespace, Name: prologJobName(tj)}
	if err := r.Get(ctx, key, &prologJob); err != nil {
		if errors.IsNotFound(err) {
			// Prolog job disappeared — recreate by going back to Pending
			return r.transition(ctx, tj, aiv1.PhasePending, "PrologMissing", "Prolog job not found, recreating")
		}
		return ctrl.Result{}, err
	}

	// Check completion
	for _, cond := range prologJob.Status.Conditions {
		switch cond.Type {
		case batchv1.JobComplete:
			if cond.Status == corev1.ConditionTrue {
				r.Recorder.Event(tj, corev1.EventTypeNormal, "PrologPassed",
					"All nodes passed GPU health check")
				tj.Status.HealthyNodes = tj.Spec.NumNodes
				tj.Status.TotalNodes = tj.Spec.NumNodes
				return r.transition(ctx, tj, aiv1.PhasePrologPassed, "HealthCheckPassed", "All nodes healthy")
			}
		case batchv1.JobFailed:
			if cond.Status == corev1.ConditionTrue {
				r.Recorder.Event(tj, corev1.EventTypeWarning, "PrologFailed",
					fmt.Sprintf("GPU health check failed: %s", cond.Message))
				tj.Status.FailureReason = "PrologFailed"
				tj.Status.FailureMessage = fmt.Sprintf("Prolog health check failed: %s", cond.Message)
				return r.transition(ctx, tj, aiv1.PhasePrologFailed, "HealthCheckFailed", cond.Message)
			}
		}
	}

	// Still running — requeue
	return ctrl.Result{RequeueAfter: prologPollInterval}, nil
}

// handlePrologPassed creates the worker StatefulSet and headless Service.
func (r *TrainJobReconciler) handlePrologPassed(ctx context.Context, tj *aiv1.TrainJob) (ctrl.Result, error) {
	// Create headless service for worker DNS resolution (needed by torch.distributed)
	svc := buildHeadlessService(tj)
	if err := controllerutil.SetControllerReference(tj, svc, r.Scheme()); err != nil {
		return ctrl.Result{}, err
	}
	if err := r.Create(ctx, svc); err != nil && !errors.IsAlreadyExists(err) {
		return ctrl.Result{}, fmt.Errorf("creating headless service: %w", err)
	}

	// Create worker StatefulSet
	sts := buildWorkerStatefulSet(tj)
	if err := controllerutil.SetControllerReference(tj, sts, r.Scheme()); err != nil {
		return ctrl.Result{}, err
	}
	if err := r.Create(ctx, sts); err != nil && !errors.IsAlreadyExists(err) {
		return ctrl.Result{}, fmt.Errorf("creating worker statefulset: %w", err)
	}

	now := metav1.Now()
	tj.Status.StartTime = &now

	r.Recorder.Event(tj, corev1.EventTypeNormal, "WorkersCreated",
		fmt.Sprintf("Created %d workers with %d GPUs each", tj.Spec.NumNodes, tj.Spec.GPUsPerNode))

	return r.transition(ctx, tj, aiv1.PhaseRunning, "WorkersLaunched",
		fmt.Sprintf("%d workers created", tj.Spec.NumNodes))
}

// handlePrologFailed is a terminal-ish state. We could add retry logic here.
func (r *TrainJobReconciler) handlePrologFailed(ctx context.Context, tj *aiv1.TrainJob) (ctrl.Result, error) {
	return r.transition(ctx, tj, aiv1.PhaseFailed, "PrologFailed",
		"Node health check failed; cannot start training")
}

// handleRunning monitors the worker StatefulSet and checks for completion or failure.
func (r *TrainJobReconciler) handleRunning(ctx context.Context, tj *aiv1.TrainJob) (ctrl.Result, error) {
	var sts appsv1.StatefulSet
	key := client.ObjectKey{Namespace: tj.Namespace, Name: workerStsName(tj)}
	if err := r.Get(ctx, key, &sts); err != nil {
		if errors.IsNotFound(err) {
			// Workers disappeared — something went very wrong
			tj.Status.FailureReason = "WorkersLost"
			tj.Status.FailureMessage = "Worker StatefulSet not found"
			return r.transition(ctx, tj, aiv1.PhaseFailed, "WorkersLost", "Worker StatefulSet disappeared")
		}
		return ctrl.Result{}, err
	}

	tj.Status.ReadyWorkers = sts.Status.ReadyReplicas

	// Check if all workers are ready
	if sts.Status.ReadyReplicas == *sts.Spec.Replicas {
		meta.SetStatusCondition(&tj.Status.Conditions, metav1.Condition{
			Type:               "AllWorkersReady",
			Status:             metav1.ConditionTrue,
			Reason:             "AllReady",
			Message:            fmt.Sprintf("%d/%d workers ready", sts.Status.ReadyReplicas, *sts.Spec.Replicas),
			LastTransitionTime: metav1.Now(),
		})
	}

	// Check for pod failures by listing pods owned by the StatefulSet
	var pods corev1.PodList
	if err := r.List(ctx, &pods,
		client.InNamespace(tj.Namespace),
		client.MatchingLabels{"training.vsr.dev/trainjob": tj.Name},
	); err != nil {
		return ctrl.Result{}, err
	}

	failedPods := 0
	for i := range pods.Items {
		if pods.Items[i].Status.Phase == corev1.PodFailed {
			failedPods++
		}
	}

	if failedPods > 0 {
		r.Recorder.Event(tj, corev1.EventTypeWarning, "WorkerFailed",
			fmt.Sprintf("%d worker pod(s) failed", failedPods))

		// If checkpointing is enabled, save checkpoint before transitioning to failed
		if tj.Spec.Checkpoint.Enabled && tj.Status.CurrentStep > 0 {
			return r.transition(ctx, tj, aiv1.PhaseCheckpointing, "WorkerFailed",
				fmt.Sprintf("%d workers failed; triggering checkpoint save", failedPods))
		}

		tj.Status.FailureReason = "WorkerPodFailed"
		tj.Status.FailureMessage = fmt.Sprintf("%d worker pod(s) failed", failedPods)
		return r.transition(ctx, tj, aiv1.PhaseFailed, "WorkerFailed",
			fmt.Sprintf("%d workers failed", failedPods))
	}

	// Check if training completed (indicated by all pods completing successfully)
	succeededPods := 0
	for i := range pods.Items {
		if pods.Items[i].Status.Phase == corev1.PodSucceeded {
			succeededPods++
		}
	}
	if int32(succeededPods) == tj.Spec.NumNodes {
		now := metav1.Now()
		tj.Status.CompletionTime = &now
		r.Recorder.Event(tj, corev1.EventTypeNormal, "TrainingCompleted",
			fmt.Sprintf("All %d workers completed successfully", tj.Spec.NumNodes))

		if tj.Spec.EvalConfig != nil {
			return r.transition(ctx, tj, aiv1.PhaseEvaluating, "TrainingCompleted",
				"All workers completed, starting post-training evaluation")
		}
		return r.transition(ctx, tj, aiv1.PhaseSucceeded, "TrainingCompleted", "All workers completed")
	}

	// Check max runtime
	if tj.Spec.MaxRuntime != nil && tj.Status.StartTime != nil {
		elapsed := time.Since(tj.Status.StartTime.Time)
		if elapsed > tj.Spec.MaxRuntime.Duration {
			tj.Status.FailureReason = "MaxRuntimeExceeded"
			tj.Status.FailureMessage = fmt.Sprintf("Job exceeded max runtime of %s", tj.Spec.MaxRuntime.Duration)
			r.Recorder.Event(tj, corev1.EventTypeWarning, "MaxRuntimeExceeded",
				tj.Status.FailureMessage)
			return r.transition(ctx, tj, aiv1.PhaseFailed, "MaxRuntimeExceeded", tj.Status.FailureMessage)
		}
	}

	// Still running — update status and requeue
	if err := r.Status().Update(ctx, tj); err != nil {
		return ctrl.Result{}, err
	}
	return ctrl.Result{RequeueAfter: runningPollInterval}, nil
}

// handleCheckpointing manages checkpoint validation after a save.
func (r *TrainJobReconciler) handleCheckpointing(ctx context.Context, tj *aiv1.TrainJob) (ctrl.Result, error) {
	if !tj.Spec.Checkpoint.ValidateOnSave {
		now := metav1.Now()
		tj.Status.LastCheckpointTime = &now
		return r.transition(ctx, tj, aiv1.PhaseRunning, "CheckpointSaved", "Checkpoint saved (validation disabled)")
	}

	valJob := buildCheckpointValidationJob(tj)
	if err := controllerutil.SetControllerReference(tj, valJob, r.Scheme()); err != nil {
		return ctrl.Result{}, err
	}
	if err := r.Create(ctx, valJob); err != nil && !errors.IsAlreadyExists(err) {
		return ctrl.Result{}, fmt.Errorf("creating checkpoint validation job: %w", err)
	}

	// Check validation job status
	var existingJob batchv1.Job
	key := client.ObjectKey{Namespace: tj.Namespace, Name: checkpointValJobName(tj)}
	if err := r.Get(ctx, key, &existingJob); err != nil {
		return ctrl.Result{RequeueAfter: checkpointPollInterval}, nil
	}

	for _, cond := range existingJob.Status.Conditions {
		if cond.Type == batchv1.JobComplete && cond.Status == corev1.ConditionTrue {
			now := metav1.Now()
			tj.Status.LastCheckpointTime = &now
			tj.Status.LastCheckpoint = fmt.Sprintf("%s/step-%d", tj.Spec.Checkpoint.StoragePath, tj.Status.CurrentStep)
			r.Recorder.Event(tj, corev1.EventTypeNormal, "CheckpointValidated",
				fmt.Sprintf("Checkpoint at step %d validated", tj.Status.CurrentStep))
			return r.transition(ctx, tj, aiv1.PhaseRunning, "CheckpointValid", "Checkpoint validated, resuming training")
		}
		if cond.Type == batchv1.JobFailed && cond.Status == corev1.ConditionTrue {
			r.Recorder.Event(tj, corev1.EventTypeWarning, "CheckpointInvalid",
				fmt.Sprintf("Checkpoint at step %d failed validation: %s", tj.Status.CurrentStep, cond.Message))
			tj.Status.FailureReason = "CheckpointValidationFailed"
			tj.Status.FailureMessage = "Checkpoint failed validation: " + cond.Message
			return r.transition(ctx, tj, aiv1.PhaseFailed, "CheckpointInvalid", cond.Message)
		}
	}

	return ctrl.Result{RequeueAfter: checkpointPollInterval}, nil
}

// handleEvaluating manages the post-training eval job.
func (r *TrainJobReconciler) handleEvaluating(ctx context.Context, tj *aiv1.TrainJob) (ctrl.Result, error) {
	evalJob := buildEvalJob(tj)
	if evalJob == nil {
		return r.transition(ctx, tj, aiv1.PhaseSucceeded, "NoEvalConfig", "No eval config, skipping evaluation")
	}

	if err := controllerutil.SetControllerReference(tj, evalJob, r.Scheme()); err != nil {
		return ctrl.Result{}, err
	}
	if err := r.Create(ctx, evalJob); err != nil && !errors.IsAlreadyExists(err) {
		return ctrl.Result{}, fmt.Errorf("creating eval job: %w", err)
	}

	var existingJob batchv1.Job
	key := client.ObjectKey{Namespace: tj.Namespace, Name: evalJobName(tj)}
	if err := r.Get(ctx, key, &existingJob); err != nil {
		return ctrl.Result{RequeueAfter: prologPollInterval}, nil
	}

	for _, cond := range existingJob.Status.Conditions {
		if cond.Type == batchv1.JobComplete && cond.Status == corev1.ConditionTrue {
			r.Recorder.Event(tj, corev1.EventTypeNormal, "EvalCompleted",
				fmt.Sprintf("Post-training evaluation completed at step %d", tj.Status.CurrentStep))
			return r.transition(ctx, tj, aiv1.PhaseSucceeded, "EvalCompleted",
				"Evaluation completed successfully")
		}
		if cond.Type == batchv1.JobFailed && cond.Status == corev1.ConditionTrue {
			r.Recorder.Event(tj, corev1.EventTypeWarning, "EvalFailed",
				fmt.Sprintf("Evaluation failed: %s", cond.Message))
			// Eval failure doesn't fail the TrainJob — training itself succeeded.
			// The failure is recorded in a condition for agents to consume.
			meta.SetStatusCondition(&tj.Status.Conditions, metav1.Condition{
				Type:               "EvalFailed",
				Status:             metav1.ConditionTrue,
				Reason:             "EvalJobFailed",
				Message:            cond.Message,
				LastTransitionTime: metav1.Now(),
			})
			return r.transition(ctx, tj, aiv1.PhaseSucceeded, "EvalFailed",
				"Training succeeded but evaluation failed: "+cond.Message)
		}
	}

	return ctrl.Result{RequeueAfter: prologPollInterval}, nil
}

// ════════════════════════════════════════════════════════════════
//  Deletion / Finalizer
// ════════════════════════════════════════════════════════════════

func (r *TrainJobReconciler) handleDeletion(ctx context.Context, tj *aiv1.TrainJob) (ctrl.Result, error) {
	logger := log.FromContext(ctx)

	if !controllerutil.ContainsFinalizer(tj, finalizerName) {
		return ctrl.Result{}, nil
	}

	// Ensure last checkpoint is preserved before we let GC clean up child resources.
	// In production, this would copy the checkpoint to a durable location.
	if tj.Status.LastCheckpoint != "" {
		logger.Info("Preserving checkpoint before deletion",
			"checkpoint", tj.Status.LastCheckpoint)
		r.Recorder.Event(tj, corev1.EventTypeNormal, "CheckpointPreserved",
			fmt.Sprintf("Preserved checkpoint %s before deletion", tj.Status.LastCheckpoint))
	}

	controllerutil.RemoveFinalizer(tj, finalizerName)
	if err := r.Update(ctx, tj); err != nil {
		return ctrl.Result{}, err
	}

	logger.Info("Finalizer removed, allowing deletion")
	return ctrl.Result{}, nil
}

// ════════════════════════════════════════════════════════════════
//  Helpers
// ════════════════════════════════════════════════════════════════

// transition updates the phase, sets a condition, and persists the status.
func (r *TrainJobReconciler) transition(
	ctx context.Context,
	tj *aiv1.TrainJob,
	phase aiv1.TrainJobPhase,
	reason, message string,
) (ctrl.Result, error) {
	logger := log.FromContext(ctx)
	logger.Info("Phase transition", "from", tj.Status.Phase, "to", phase, "reason", reason)

	tj.Status.Phase = phase
	meta.SetStatusCondition(&tj.Status.Conditions, metav1.Condition{
		Type:               string(phase),
		Status:             metav1.ConditionTrue,
		Reason:             reason,
		Message:            message,
		LastTransitionTime: metav1.Now(),
	})

	if err := r.Status().Update(ctx, tj); err != nil {
		return ctrl.Result{}, fmt.Errorf("updating status to %s: %w", phase, err)
	}

	// Immediately requeue to process the new phase (except terminal states)
	if phase == aiv1.PhaseSucceeded || phase == aiv1.PhaseFailed {
		return ctrl.Result{}, nil
	}
	return ctrl.Result{Requeue: true}, nil
}

func prologJobName(tj *aiv1.TrainJob) string {
	return tj.Name + "-prolog"
}

func workerStsName(tj *aiv1.TrainJob) string {
	return tj.Name + "-workers"
}

func headlessSvcName(tj *aiv1.TrainJob) string {
	return tj.Name + "-headless"
}

func checkpointValJobName(tj *aiv1.TrainJob) string {
	return fmt.Sprintf("%s-ckpt-val-%d", tj.Name, tj.Status.CurrentStep)
}

// ════════════════════════════════════════════════════════════════
//  Controller setup
// ════════════════════════════════════════════════════════════════

func (r *TrainJobReconciler) SetupWithManager(mgr ctrl.Manager) error {
	return ctrl.NewControllerManagedBy(mgr).
		For(&aiv1.TrainJob{}).
		Owns(&batchv1.Job{}).        // Prolog + checkpoint validation jobs
		Owns(&appsv1.StatefulSet{}). // Worker StatefulSet
		Owns(&corev1.Service{}).     // Headless service
		Complete(r)
}
