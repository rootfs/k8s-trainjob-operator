package main

import (
	"flag"
	"os"

	"k8s.io/apimachinery/pkg/runtime"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	clientgoscheme "k8s.io/client-go/kubernetes/scheme"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/healthz"
	"sigs.k8s.io/controller-runtime/pkg/log/zap"
	metricsserver "sigs.k8s.io/controller-runtime/pkg/metrics/server"
	ctrlwebhook "sigs.k8s.io/controller-runtime/pkg/webhook"

	aiv1 "github.com/rootfs/trainjob-operator/api/v1alpha1"
	"github.com/rootfs/trainjob-operator/internal/controller"
	"github.com/rootfs/trainjob-operator/internal/webhook"
)

var scheme = runtime.NewScheme()

func init() {
	utilruntime.Must(clientgoscheme.AddToScheme(scheme))
	utilruntime.Must(aiv1.AddToScheme(scheme))
}

func main() {
	var metricsAddr string
	var probeAddr string
	var enableLeaderElection bool
	var webhookPort int

	flag.StringVar(&metricsAddr, "metrics-bind-address", ":8080", "The address the metric endpoint binds to.")
	flag.StringVar(&probeAddr, "health-probe-bind-address", ":8081", "The address the probe endpoint binds to.")
	flag.BoolVar(&enableLeaderElection, "leader-elect", false,
		"Enable leader election for controller manager, ensuring only one active controller.")
	flag.IntVar(&webhookPort, "webhook-port", 9443, "The port the webhook server binds to.")
	flag.Parse()

	ctrl.SetLogger(zap.New(zap.UseDevMode(true)))
	logger := ctrl.Log.WithName("setup")

	mgr, err := ctrl.NewManager(ctrl.GetConfigOrDie(), ctrl.Options{
		Scheme: scheme,
		Metrics: metricsserver.Options{
			BindAddress: metricsAddr,
		},
		WebhookServer: ctrlwebhook.NewServer(ctrlwebhook.Options{
			Port: webhookPort,
		}),
		HealthProbeBindAddress: probeAddr,
		LeaderElection:         enableLeaderElection,
		LeaderElectionID:       "trainjob-operator.training.vsr.dev",
	})
	if err != nil {
		logger.Error(err, "unable to create manager")
		os.Exit(1)
	}

	// ── Register the TrainJob reconciler ──
	if err := (&controller.TrainJobReconciler{
		Client:   mgr.GetClient(),
		Recorder: mgr.GetEventRecorderFor("trainjob-controller"),
	}).SetupWithManager(mgr); err != nil {
		logger.Error(err, "unable to create controller", "controller", "TrainJob")
		os.Exit(1)
	}

	// ── Register webhooks ──

	// Validating webhook for TrainJob
	if err := ctrl.NewWebhookManagedBy(mgr).
		For(&aiv1.TrainJob{}).
		WithValidator(&webhook.TrainJobValidator{}).
		Complete(); err != nil {
		logger.Error(err, "unable to create validating webhook", "webhook", "TrainJob")
		os.Exit(1)
	}

	// Mutating webhook for TrainJob (NCCL injection, defaults)
	if err := ctrl.NewWebhookManagedBy(mgr).
		For(&aiv1.TrainJob{}).
		WithDefaulter(&webhook.TrainJobMutator{}).
		Complete(); err != nil {
		logger.Error(err, "unable to create mutating webhook", "webhook", "TrainJob")
		os.Exit(1)
	}

	// Note: The GPU monitoring sidecar is now included directly in the
	// worker StatefulSet pod template by the reconciler (workers.go),
	// eliminating the need for a cluster-wide pod mutating webhook.
	// The PodSidecarInjector in internal/webhook/ is retained as a
	// legacy option for clusters that prefer webhook-based injection.

	// ── Health checks ──
	if err := mgr.AddHealthzCheck("healthz", healthz.Ping); err != nil {
		logger.Error(err, "unable to set up health check")
		os.Exit(1)
	}
	if err := mgr.AddReadyzCheck("readyz", healthz.Ping); err != nil {
		logger.Error(err, "unable to set up ready check")
		os.Exit(1)
	}

	// ── Start ──
	logger.Info("Starting manager",
		"metrics", metricsAddr,
		"webhooks", webhookPort,
		"leader-election", enableLeaderElection,
	)
	if err := mgr.Start(ctrl.SetupSignalHandler()); err != nil {
		logger.Error(err, "problem running manager")
		os.Exit(1)
	}
}
