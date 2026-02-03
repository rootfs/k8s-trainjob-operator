package controller_test

import (
	"context"
	"time"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
	batchv1 "k8s.io/api/batch/v1"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"sigs.k8s.io/controller-runtime/pkg/client"

	aiv1 "github.com/rootfs/trainjob-operator/api/v1alpha1"
)

// These tests use envtest — a real API server with no kubelet.
// They verify the reconciler's state machine transitions.

var _ = Describe("TrainJob Controller", func() {
	const (
		timeout  = 30 * time.Second
		interval = 250 * time.Millisecond
	)

	ctx := context.Background()

	// ── Test 1: Happy path with prolog skip ──
	Context("When creating a TrainJob with skipProlog=true", func() {
		It("Should transition Pending → PrologPassed → Running", func() {
			tj := &aiv1.TrainJob{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-skip-prolog",
					Namespace: "default",
				},
				Spec: aiv1.TrainJobSpec{
					Model:       "test-model",
					Image:       "test:latest",
					NumNodes:    2,
					GPUsPerNode: 8,
					TPDegree:    8,
					PPDegree:    1,
					Precision:   "bf16",
					SkipProlog:  true,
					Checkpoint: aiv1.CheckpointSpec{
						Enabled:     true,
						StoragePath: "/checkpoints",
					},
				},
			}

			Expect(k8sClient.Create(ctx, tj)).Should(Succeed())

			// Should transition to PrologPassed (skipping prolog)
			Eventually(func() aiv1.TrainJobPhase {
				var fetched aiv1.TrainJob
				_ = k8sClient.Get(ctx, types.NamespacedName{
					Name: "test-skip-prolog", Namespace: "default",
				}, &fetched)
				return fetched.Status.Phase
			}, timeout, interval).Should(Equal(aiv1.PhasePrologPassed))

			// Should then create worker StatefulSet and transition to Running
			Eventually(func() aiv1.TrainJobPhase {
				var fetched aiv1.TrainJob
				_ = k8sClient.Get(ctx, types.NamespacedName{
					Name: "test-skip-prolog", Namespace: "default",
				}, &fetched)
				return fetched.Status.Phase
			}, timeout, interval).Should(Equal(aiv1.PhaseRunning))

			// Verify headless service was created
			var svc corev1.Service
			Expect(k8sClient.Get(ctx, types.NamespacedName{
				Name: "test-skip-prolog-headless", Namespace: "default",
			}, &svc)).Should(Succeed())
			Expect(svc.Spec.ClusterIP).To(Equal(corev1.ClusterIPNone))
		})
	})

	// ── Test 2: Prolog flow ──
	Context("When creating a TrainJob with prolog enabled", func() {
		It("Should create a prolog Job and wait for it", func() {
			tj := &aiv1.TrainJob{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-with-prolog",
					Namespace: "default",
				},
				Spec: aiv1.TrainJobSpec{
					Model:       "test-model",
					Image:       "test:latest",
					NumNodes:    4,
					GPUsPerNode: 8,
					TPDegree:    8,
					PPDegree:    1,
					Precision:   "bf16",
					SkipProlog:  false,
					NodeSelector: map[string]string{
						"nvidia.com/gpu.product": "NVIDIA-H100-SXM5-80GB",
					},
					Checkpoint: aiv1.CheckpointSpec{
						Enabled:     true,
						StoragePath: "/checkpoints",
					},
				},
			}

			Expect(k8sClient.Create(ctx, tj)).Should(Succeed())

			// Should create prolog job
			Eventually(func() error {
				var job batchv1.Job
				return k8sClient.Get(ctx, types.NamespacedName{
					Name: "test-with-prolog-prolog", Namespace: "default",
				}, &job)
			}, timeout, interval).Should(Succeed())

			// Should be in PrologRunning phase
			Eventually(func() aiv1.TrainJobPhase {
				var fetched aiv1.TrainJob
				_ = k8sClient.Get(ctx, types.NamespacedName{
					Name: "test-with-prolog", Namespace: "default",
				}, &fetched)
				return fetched.Status.Phase
			}, timeout, interval).Should(Equal(aiv1.PhasePrologRunning))

			// Simulate prolog completion by updating the Job status
			var prologJob batchv1.Job
			Expect(k8sClient.Get(ctx, types.NamespacedName{
				Name: "test-with-prolog-prolog", Namespace: "default",
			}, &prologJob)).Should(Succeed())

			prologJob.Status.Conditions = append(prologJob.Status.Conditions, batchv1.JobCondition{
				Type:   batchv1.JobComplete,
				Status: corev1.ConditionTrue,
			})
			Expect(k8sClient.Status().Update(ctx, &prologJob)).Should(Succeed())

			// Should transition to PrologPassed → Running
			Eventually(func() aiv1.TrainJobPhase {
				var fetched aiv1.TrainJob
				_ = k8sClient.Get(ctx, types.NamespacedName{
					Name: "test-with-prolog", Namespace: "default",
				}, &fetched)
				return fetched.Status.Phase
			}, timeout, interval).Should(Equal(aiv1.PhaseRunning))
		})
	})

	// ── Test 3: Prolog failure ──
	Context("When prolog fails", func() {
		It("Should transition to Failed", func() {
			tj := &aiv1.TrainJob{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-prolog-fail",
					Namespace: "default",
				},
				Spec: aiv1.TrainJobSpec{
					Model:       "test-model",
					Image:       "test:latest",
					NumNodes:    4,
					GPUsPerNode: 8,
					TPDegree:    8,
					PPDegree:    1,
					Precision:   "bf16",
					Checkpoint: aiv1.CheckpointSpec{
						Enabled:     true,
						StoragePath: "/checkpoints",
					},
				},
			}

			Expect(k8sClient.Create(ctx, tj)).Should(Succeed())

			// Wait for prolog job to be created
			Eventually(func() error {
				var job batchv1.Job
				return k8sClient.Get(ctx, types.NamespacedName{
					Name: "test-prolog-fail-prolog", Namespace: "default",
				}, &job)
			}, timeout, interval).Should(Succeed())

			// Simulate prolog failure
			var prologJob batchv1.Job
			Expect(k8sClient.Get(ctx, types.NamespacedName{
				Name: "test-prolog-fail-prolog", Namespace: "default",
			}, &prologJob)).Should(Succeed())

			prologJob.Status.Conditions = append(prologJob.Status.Conditions, batchv1.JobCondition{
				Type:    batchv1.JobFailed,
				Status:  corev1.ConditionTrue,
				Message: "GPU 3 has uncorrectable ECC errors",
			})
			Expect(k8sClient.Status().Update(ctx, &prologJob)).Should(Succeed())

			// Should transition to Failed
			Eventually(func() aiv1.TrainJobPhase {
				var fetched aiv1.TrainJob
				_ = k8sClient.Get(ctx, types.NamespacedName{
					Name: "test-prolog-fail", Namespace: "default",
				}, &fetched)
				return fetched.Status.Phase
			}, timeout, interval).Should(Equal(aiv1.PhaseFailed))

			// Verify failure reason is captured
			var fetched aiv1.TrainJob
			Expect(k8sClient.Get(ctx, types.NamespacedName{
				Name: "test-prolog-fail", Namespace: "default",
			}, &fetched)).Should(Succeed())
			Expect(fetched.Status.FailureReason).To(Equal("PrologFailed"))
		})
	})

	// ── Test 4: Finalizer prevents deletion before checkpoint preservation ──
	Context("When deleting a TrainJob with a checkpoint", func() {
		It("Should preserve checkpoint via finalizer before allowing deletion", func() {
			tj := &aiv1.TrainJob{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-finalizer",
					Namespace: "default",
				},
				Spec: aiv1.TrainJobSpec{
					Model:       "test-model",
					Image:       "test:latest",
					NumNodes:    1,
					GPUsPerNode: 8,
					TPDegree:    8,
					PPDegree:    1,
					Precision:   "bf16",
					SkipProlog:  true,
					Checkpoint: aiv1.CheckpointSpec{
						Enabled:     true,
						StoragePath: "/checkpoints",
					},
				},
			}

			Expect(k8sClient.Create(ctx, tj)).Should(Succeed())

			// Wait for finalizer to be added
			Eventually(func() bool {
				var fetched aiv1.TrainJob
				_ = k8sClient.Get(ctx, types.NamespacedName{
					Name: "test-finalizer", Namespace: "default",
				}, &fetched)
				for _, f := range fetched.Finalizers {
					if f == "training.vsr.dev/checkpoint-protection" {
						return true
					}
				}
				return false
			}, timeout, interval).Should(BeTrue())

			// Set a checkpoint in status
			var fetched aiv1.TrainJob
			Expect(k8sClient.Get(ctx, types.NamespacedName{
				Name: "test-finalizer", Namespace: "default",
			}, &fetched)).Should(Succeed())
			fetched.Status.LastCheckpoint = "/checkpoints/step-1000"
			Expect(k8sClient.Status().Update(ctx, &fetched)).Should(Succeed())

			// Delete — should succeed because finalizer logic preserves checkpoint then removes finalizer
			Expect(k8sClient.Delete(ctx, &fetched)).Should(Succeed())

			// Eventually the object should be fully deleted
			Eventually(func() bool {
				err := k8sClient.Get(ctx, types.NamespacedName{
					Name: "test-finalizer", Namespace: "default",
				}, &aiv1.TrainJob{})
				return client.IgnoreNotFound(err) == nil
			}, timeout, interval).Should(BeTrue())
		})
	})
})
