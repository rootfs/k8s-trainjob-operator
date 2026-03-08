# Kueue Integration

The operator supports coexistence with [Kueue](https://github.com/kubernetes-sigs/kueue) via a **suspend-based admission pattern**. This is the same pattern Kueue uses for batch Jobs: the workload starts suspended, Kueue holds it until quota is available, then unsuspends it to start.

## How it works

1. **User submits a TrainJob** with a `kueue.x-k8s.io/queue-name` label pointing to a Kueue LocalQueue.
2. **Mutating webhook** detects the label and auto-sets `spec.suspend: true` (if not already set by the user).
3. **Reconciler** sees `suspend=true` and enters the `Suspended` phase — no prolog Job, no StatefulSet, no headless Service. The TrainJob CR exists but produces no child resources.
4. **Kueue** sees the TrainJob workload (via a Kueue integration or generic job webhook). When quota is granted, Kueue sets `spec.suspend: false`.
5. **Reconciler** detects the unsuspend, transitions to `Pending`, and proceeds normally: prolog → workers → running.

## Child resource isolation

The prolog Job, checkpoint validation Job, and worker StatefulSet are created by the reconciler — they're not user-submitted. Kueue's webhook intercepts Jobs by default (and in some configurations, all pods), which would cause it to try to manage these internal resources.

To prevent this, all child Jobs carry a `kueue.x-k8s.io/queue-name: none` label. This signals to Kueue that these are operator-managed internal resources and should not be admitted through Kueue's queue. The TrainJob itself is the unit of admission — its children inherit its quota grant implicitly via the suspend/unsuspend handshake.

## Without Kueue

If no `kueue.x-k8s.io/queue-name` label is present, `spec.suspend` stays nil (falsy) and the reconciler starts immediately. This is the standalone mode for dedicated clusters without a queueing system.

## What this doesn't solve

- **Gang scheduling**: Kueue provides admission control (when to start), not gang scheduling (ensure all pods are co-scheduled). The operator still relies on StatefulSet pod management, which is best-effort. For true gang scheduling, you'd need Kueue + a scheduler plugin or Volcano.
- **TAS (Topology-Aware Scheduling)**: Kueue's TAS can place the TrainJob's pods with rack/switch awareness, but the operator doesn't set the topology annotations. This is a gap — you'd configure TAS at the Kueue ClusterQueue level.
- **Webhook stacking**: If Kueue, the TrainJob operator, and other controllers all register mutating webhooks for overlapping resource types, the kube-apiserver runs them serially. The operator minimizes this by only registering webhooks for TrainJob resources (not Jobs or Pods), but the interaction between Kueue's Job webhook and the operator's child Jobs is why the `kueue.x-k8s.io/queue-name: none` label exists.

## Example

See `examples/trainjob_sample.yaml` (Example 8) for a complete Kueue-integrated TrainJob.
