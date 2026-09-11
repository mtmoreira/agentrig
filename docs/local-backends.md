# Replaceable local backends

The local SDK candidate `0.3.1.dev6` adds `LocalBackend` in `capabilities`,
separate from model-facing `AgentRuntime`. No Python dependency or runtime import
is added. The published `0.3.0` artifact is unchanged.

Applications construct an explicit `LocalBackendCatalog`. Resolution requires
the requested capabilities. Unknown IDs, changed descriptors and incompatible
capabilities fail before invocation; there is no discovery or dynamic import.
Adapters own configuration keys. Plans snapshot inputs, capabilities, version,
operation and effects under one SHA-256 digest. Review exports are detached.

`BackendAuthority` is supplied for one invocation; saved state cannot restore
permission. Adapters repeat preflight before effects. The catalog checks returned
operation, plan digest and action-compatible status. Cancellation and deadlines
remain native signals; unexpected adapter failures become fixed safe codes.
This contract does not itself provide runtime ownership, a journal or retries.

## Current adapters and limits

`ExistingRootlessDockerBackend` inspects one explicit endpoint through an injected
verifier. That verifier owns executable, current-user socket, rootless daemon,
architecture and transport checks. Without a verifier the plan is blocked. It
cannot create a VM, acquire an image or run a workload.

`LimaBackend` produces a **review-only** Lima 2.2.0 VZ preparation proposal for
Darwin arm64. It advertises `local-vm-plan.v1`, not a Docker execution endpoint.
Its fixed blockers cannot be cleared by hashes, flags or fabricated receipts.
`apply` always fails closed. There is no installer or lifecycle driver.

The proposal has no host mounts, inherited templates, provision scripts,
containerd installation, proxy propagation, ambient SSH key loading or automatic
application TCP/UDP forwarding. It is deliberately incomplete and must not be
passed to Lima as executable configuration. Still required:

- Review complete guest image, rootless engine, package and license inventories.
- Verify archive bytes/members, executable pins and immutable guest bytes.
- Suppress default guest dependency installation and verify cgroup delegation.
- Implement private Unix forwarding and owned staging with explicit guest paths.
- Implement bounded process handling, durable intent and exact-instance recovery;
  never automatically replay uncertain creation or startup.
- Validate the real VM, isolation, resources and recovery with explicit authority.

`CommandTool` now uses `run_bounded_process` to drain output with fixed capture
memory. Its compatible truncation mode retains at most two configured bounds
while reading and returns one combined bound. `ProcessSpec` defaults to rejecting
overflow, as required by the Lima lifecycle driver. Exact argv, private cwd,
explicit environment, deadlines and cancellation apply without a shell or retries.
An unreaped child anchors process-group cleanup even when its descendants hold
pipes open. Native cancellation waits for worker cleanup. This is not an operating
system sandbox: trusted executables can deliberately detach other processes.

## Durable lifecycle candidate

`JournaledLocalBackends` writes an uncertain intent before calling an adapter.
Exceptions, cancellation and failed completion writes retain uncertainty. Each
operation ID is single-use. Reconciliation only performs a fresh authorized
`INSPECT` for the same backend, version, instance configuration and operation;
it never replays create/start/stop or restores a saved grant.

`LocalBackendJournal` uses a caller-supplied, ancestor-verified private directory,
descriptor-relative bounded JSON, file and directory fsync, an exclusive writer
lock, and local directory identity bindings. It rejects copied state, links,
public files and concurrent writers. Interrupted staging files are preserved and
block new writes; they require explicit repair, not automatic deletion. `inspect`
is a read-only diagnostic snapshot. Execution uses the locked current state.

`LimaLifecycleBackend` implements exact create/start/stop/list command sequences
and journaled recovery. It is separate from the default review-only `LimaBackend`
and is not registered automatically. It requires an injected **trusted application
artifact verifier**, not a user-configurable readiness boolean. Before every
command, that verifier must verify the approved complete local artifact closure,
executable/configuration digests, private roots, absence of global overrides,
owned instance identity and suppression of all implicit acquisition. Without it,
every plan is blocked. No production verifier or approved guest bundle exists yet.

The driver uses a private LIMA_HOME/HOME and exact instance name; never enumerate
another account's runtime or use a default instance. VM observations must match
that name, directory, architecture and VM type. Start requires observed stopped
state; prepare requires local absence; stop requires observed running state.
Unknown results leave uncertainty. Even a running VM result does not qualify
rootless Docker, cgroups, guest-visible staging or simulation readiness.

`WorkspaceStagePlan` snapshots bounded files and binds approval to exact host and
guest roots, operation identity and file hashes. `stage_workspace` writes only a
new private operation directory using exclusive descriptor-relative creation,
fsync and intent/completion receipts. Existing directories, links, traversal,
path conflicts and oversized input are refused; partial files are retained.
`LimaLifecycleBackend.stage` fixes the mapping to its private staging root and
`/mnt/agentrig`, and refuses staging while lifecycle recovery is unresolved.
The returned guest path is a proposed mapping, **not verified guest visibility**.
An approved configuration verifier and real mount/Unix-socket qualification are
still required before a workload can use it.

Offline subprocess and filesystem integration tests exercise the real local
process runner and journal. Lima command tests use an injected scripted process;
they are **not** evidence that Lima, a VM, or a simulator was run.

## Local artifact audit

`BackendArtifactManifest` declares exact IDs, relative paths, SHA-256 hashes,
sizes and executable flags. `audit_backend_artifacts` reads one caller-supplied,
ancestor-verified private bundle directory with a mandatory finite deadline.
It inventories the exact directory tree before opening payloads, rejects extra
files, links, hardlinks and non-private modes, hashes in 64 KiB chunks, and checks
file/directory identity before returning. Nothing is downloaded, extracted,
executed, installed or written. Declare only reviewed public runtime artifacts;
hidden/authentication filenames and private-key/certificate suffixes are refused.

Limits are 128 files, depth 8, 4 GiB per file and 16 GiB total. Files must have
mode 0600 (data) or 0700 (executable), with private directories. A result verifies
the declared bytes at that point in time; it does not prove that the manifest
contains all guest dependencies, that the executable is safe, or that an archive
has safe members. Reverify before later use. The audit is deliberately not a
`LimaArtifactVerifier`: clearing the lifecycle gate still needs semantic config,
complete artifact/dependency closure, owned-instance and no-acquisition checks.

## Guest workspace transfer candidate

`copy_to_guest` connects once to a `GuestWorkspaceEndpoint` supplied by the
selected backend. Approval binds the plan, instance, private directory and exact
socket device/inode/owner. Only `workspace.sock` under that directory is allowed;
it must be a 0600 socket owned by the current user, with one link. Directory and
socket identity are checked before connection, before sending, and after the
receipt. Missing or changed endpoints fail without reconnecting. Ancestor trust
and VM forwarding authentication remain the backend's responsibility; local
socket ownership alone is not proof of the guest peer's identity.

`transfer_guest_workspace` and `receive_guest_workspace` implement the same
`agentrig.guest-workspace.v1` framed protocol over a caller-supplied, connected
Unix stream. A backend must first authenticate and bind that connection to its
owned instance, journal fresh operation authority, and verify any VM socket
forward. The functions consume and close the connection. They never discover an
endpoint, start a listener or VM, execute collateral, or contact Docker.

The sender requires the exact `WorkspaceStagePlan.digest`, sends a bounded JSON
header and raw file bytes, then half-closes its write side. The receiver checks
the instance, fixed guest root, unique operation, path allowlist, file sizes and
hashes before acknowledging. Both require a context deadline; socket waits poll
cancellation and have a 60-second wall-clock ceiling. Headers are at most 64 KiB;
there are at most 128 files, 1 MiB per file, 16 MiB total. Files must be under
explicit subdirectories; root files and known credential/key paths are refused.
Only deliberately supplied bytes are transmitted; no host file is opened.

The guest operation parent stays 0700. Its intent and completion records are
0600 and contain only digests, counts and local filesystem identity. The copied
subdirectories are 0755 and files 0644, inside that private parent, so an executor
can bind only those subdirectories read-only into a differently mapped workload
UID. Files are reopened and rehashed before completion. Existing operation
directories always fail, including partial transfers. Disconnection or timeout
may leave completed or partial state; callers must retain uncertainty and must
not reconnect/replay automatically. A response is an observation, never authority.

`workload_access_verified` is permanently false in this receipt. Local socket
tests establish copy/validation behavior only. The functions are synchronous;
an async adapter must own and join its worker through cancellation. Production
service activation, peer authentication, endpoint forwarding, recovery inspection,
and actual container mount/UID qualification remain activation gates. The default
backend catalog does not register this candidate. No new Python dependency was
added; guest transfer was introduced in 0.3.1.dev4 and public 0.3.0 remains unchanged.

The planned OpenRTL integration will map its reviewed `input/` and `control/`
files to these guest subdirectories, preserving read-only binds, UID 65534, no network,
resource limits and its existing evidence checks. Host staging can remain
0700/0600. A replacement backend can supply the same authenticated stream without
any Lima branch in the engineering state machine.

## Guest dependency and activation checklist

This source helper does not make a guest image a verified offline closure. Before
activation, review immutable bytes and licensing for the guest OS, its Python
interpreter/standard library, this exact SDK/helper build, the rootless engine
and its required user-namespace/network/resource helpers, the simulation image,
and the Lima host/guest agents and their execution dependencies. The bundle must
also provide the reviewed service definition, dedicated user/private state,
local Unix socket and exact forwarding configuration. No runtime acquisition is
performed by either transfer function.

The artifact audit verifies only declared files. A production verifier must also
check the effective Lima configuration (including absence of global overrides),
no stock-template fallback or implicit acquisition, exact local image and helper
pins, no host-home mount, and only the owned socket forwards. Actual service peer
identity, VM forwarding, rootless cgroups and workload read-only access need live
qualification. Missing pins or observations keep readiness false.

Lima 2.2.0's [shell implementation](https://github.com/lima-vm/lima/blob/v2.2.0/cmd/limactl/shell.go)
constructs an SSH command invoking a guest login shell. This candidate does not
use `limactl shell`, `--sync`, SSH argv, shell startup files or an installer hook
as its file-transfer mechanism. The preinstalled guest helper and its owned
forwarded socket must be separately reviewed before any real invocation.

## Restricted Lima input configuration

`LimaConfiguration` renders one complete JSON input profile for Lima 2.2.0.
It validates resource bounds, a literal private state root, dedicated guest UID,
and the exact local guest image pin and size. `document()` returns a detached
mapping; `validate()` accepts only that exact semantic profile, including JSON
types. Unknown fields, duplicate keys, YAML, templates, fallback images, arbitrary
provisioning and changes to forwarding, mounts or dependency policy fail closed.

The profile disables containerd installation, host-home mounts, copying back to
the host, proxy propagation, host resolver integration, Rosetta, package upgrades,
ambient SSH keys and agent/X11 forwarding. It permits only the owned workspace
and Docker Unix-socket forwards and ignores application TCP/UDP forwarding.
The dedicated guest account retains Lima's passwordless-sudo boot assumption;
that is not permission for an OpenRTL container to run as the guest account.
The workload must still use UID 65534 and pass independent rootless/isolation
checks. The VM's default NAT is not a network isolation guarantee.

The single fixed dependency provision is a no-op with
`skipDefaultDependencyResolution: true`. Lima's
[2.2.0 dependency boot script](https://github.com/lima-vm/lima/blob/v2.2.0/pkg/cidata/cidata.TEMPLATE.d/boot.Linux/30-install-packages.sh)
otherwise installs missing packages even with containerd disabled. This renderer
does not execute that script. Preinstalled guest software and OS update services
must be reviewed as part of the complete image; suppressing Lima's installer
does not prove that the image has all dependencies or never acquires software.

`audit_lima_configuration` checks existing private `configuration.json` and
`artifacts/guest.raw`, the exact image bytes, and absence of global Lima
configuration. `lima/_config` may be absent or empty/private; any entry is refused
without opening it. It rechecks identities after reading, never reads instance
keys or contacts a socket, and does not write or select a runtime. Its caller must
verify root ancestors. Image files use the existing bounded artifact audit.

This audit verifies input semantics and declared bytes. It deliberately leaves
`effective_configuration_verified`, `dependency_closure_qualified` and
`execution_authorized` false. Lima may materialize defaults and instance YAML;
those effective settings, the host binaries/guest agent, the service/helper,
license/dependency inventory and real forwarding remain separate gates. This
function is not a complete `LimaArtifactVerifier` and is not registered as one.
No Lima binary has parsed the profile yet. Candidate parser/runtime qualification
needs the separately approved artifact acquisition and execution step.

## Journaled transfer recovery

`JournaledGuestWorkspaces` coordinates one copy under a dedicated per-instance
transfer journal, separate from the VM lifecycle journal. `guest_operation_digest`
binds each fresh grant to copy versus inspection, the observed endpoint identity,
instance, operation, paths and exact file bytes. Journal writes require explicit
invocation authority. With no trusted endpoint verifier, execution is blocked.

The coordinator holds an exclusive writer lease, persists uncertainty before
calling the backend verifier or sending bytes, and repeats verification after
the reply. The verifier must authenticate the exact VM peer and forwarding,
honor the context, return `None` on success and raise on failure. It is an
application-supplied adapter, never decoded from a readiness flag or saved state.
The copier is replaceable; the default uses the existing private Unix connector.

Failures, cancellation, bad receipts and completion-write failures retain
uncertainty. The same operation cannot be resent. Recovery requires a fresh
`inspect` grant and a separately supplied authenticated read-only guest inspector.
The coordinator compares every receipt field before settling. The generic
journal's `PREPARE`/`prepared` values mean workspace preparation in this dedicated
namespace; they never describe VM state. Successful inspection settles a copy as
completed and does not restore permission or enable workload execution.

`inspect_guest_workspace` runs inside the guest service against its existing
ancestor-verified root. It checks both private receipts, their local directory
identity, the exact bounded inventory, file hashes, modes and link counts, then
rechecks identities. Extra files, missing completion, corruption, copied receipts
and changed roots fail. It performs no writes, repairs, cleanup, sockets or replay.
Missing state is not treated as safe absence. The method requires a finite deadline.

Tests exercise real local journals and socketpair copies with synthetic endpoint
authentication. They do not validate a VM peer or a deployed service. No guest
inspection RPC, listener, service activation, effective Lima parser or complete
artifact verifier is registered by default. Those gates and actual rootless workload
visibility remain pending; `workload_access_verified` stays false.

## Bounded guest service candidate

`handle_guest_request` consumes one authenticated Unix stream and closes it on
every exit. The versioned dispatch frame selects only `copy` or `inspect` and
binds the instance. Copy uses the existing bounded workspace protocol. Inspection
sends only a manifest, requires EOF before reading guest files, and hashes the
existing workspace and receipts without writes or replay. The client accepts only
the exact expected receipt and EOF. Deadlines and cancellation remain mandatory.

`copy_guest_service` and `inspect_guest_service` fit the journal's replaceable
transport ports. The journal still requires a fresh action/endpoint-bound grant
and backend verification before and after transport. `request_guest_service`
is the lower-level owned-stream API; its plan digest alone does not authenticate
the outer VM connection or authorize a journal action.

The default peer verifier uses Linux `SO_PEERCRED` before any request reads,
requiring the dedicated service account's UID and rejecting root, UID 65534,
unsupported platforms and invalid kernel metadata. Per the
[Linux Unix socket contract](https://man7.org/linux/man-pages/man7/unix.7.html),
these credentials identify the connected guest-local process at connection time.
They do not identify the original caller across forwarding. Kernel IDs are never
retained. The VM and host-forwarding verifier remains a required backend adapter;
an injected guest verifier is trusted application code, not request configuration.

`serve_guest_once` is an explicit, one-request listener function, not an installed
daemon or customer setup command. It anchors existing private directories, binds
`workspace.sock` with mode 0600 at creation, verifies identities and closes after
one request. Its temporary restrictive process umask requires caller-owned process
isolation. Existing entries block activation and are never adopted or removed.
The socket entry is deliberately retained after close for the client's final
identity check and evidence. Reviewed backend-owned cleanup/reactivation and a
long-lived service lifecycle remain pending. Waiting is bounded by the context
and a 60-second wall-clock ceiling.

Real local socketpair and filesystem-listener tests use synthetic peer verification;
unit tests supply synthetic Linux credential metadata. Neither proves actual Linux
kernel authentication, VM forwarding, service deployment or container visibility.
No service, Lima backend or workload is activated by importing these APIs. Actual
Linux peer checks and deployed service qualification remain acceptance gates.

## Bounded service lifecycle and retirement

The guest-service lifecycle now supports `serve_guest_requests` for 1..64
sequential requests on one private endpoint. `serve_guest_once` delegates with
a limit of one. Each connection is independently authenticated, and the first
request failure stops the service. Admission lasts at most 60 seconds; an
in-flight request keeps its transport deadline. A finite context deadline is
always required. There is no automatic retry, restart, daemon or installation.

A persistent empty `.service.lock` file holds an exclusive local lease across
the listener lifetime. Its inode, mode and ownership are rechecked alongside the
directories and socket. Concurrent cooperating service/retirement operations are
blocked. Close, cancellation and failure release descriptors but retain the socket,
lock and workspace evidence. A free lease does not prove a VM service is stopped.

`guest_service_retirement_digest` binds a fresh cleanup review to the exact root,
instance and socket identity. `retire_guest_service` requires that grant and a
trusted backend quiescence verifier. The verifier must establish that the exact
service and forwarding are stopped and clients have finished their final endpoint
checks. Nothing selects this verifier from disk or request data. Under the lease,
the function rechecks state after verification and removes only `workspace.sock`,
then syncs the directory. The lock and all workspace artifacts remain. Missing,
replaced or non-socket entries fail; old one-shot entries without the lock are
not adopted. A verified retirement permits a later explicitly authorized bind.

This is a low-level retirement port. The application must persist operation
uncertainty before invocation through its backend coordinator. A lost reply or
post-unlink failure requires fresh reconciliation; absence never restores cleanup
or runtime authority. Crash recovery additionally requires backend proof of stopped
service/forwarding and drained clients. The production quiescence verifier and
backend registration remain pending. Tests exercise local leases and listener
copy/inspect/retire/rebind with synthetic authentication/quiescence, not a VM.

## Journaled retirement recovery

`JournaledGuestServiceRetirement` uses a dedicated per-instance journal separate
from workspace copies and VM lifecycle operations. Review
`guest_retirement_operation_digest(endpoint, operation_id, action)` with action
`retire` or `inspect`; these approvals are distinct and bind the exact endpoint
and operation. Explicit journal-write authority is also required. Neither stored
intent nor a previous grant authorizes a new call.

The replaceable `GuestServiceRetirementAdapter` owns backend authentication,
shutdown verification and transport. Its `quiescent` context must authenticate
the exact service generation, prove forwarding stopped and clients drained, and
prevent restart/replacement throughout retirement and observation. A free local
lease, saved PID or absent socket cannot establish this. `retire` uses the guarded
socket-retirement port or its authenticated guest equivalent. `inspect_retired`
must freshly verify the owned namespace and retained evidence for that exact
endpoint; missing, unavailable or replaced state alone must fail. These trusted
methods return exactly `None` on success and honor cancellation/deadlines.

The coordinator persists uncertainty before entering that backend context, then
performs removal, read-only verification and context exit before settling the
journal. Lost replies, invalid results, context failures and failed publication
leave uncertainty. A separately approved `inspect` call can reconcile only the
exact original intent; it never invokes removal. Uncertain operations block new
retirements, and reused operation IDs are rejected. Journal `stopped` refers only
to the observed service retirement, never proof of VM shutdown or ongoing runtime
authority. A successful record is a historical observation.

No production adapter is selected by default, and this API installs or starts
nothing. Tests exercise real durable journals with synthetic backend observations.
Production Linux/VM authentication, restart exclusion and retirement observation
still require independent qualification. Customer clone/run setup is unchanged.

## Registering another backend

Implement the protocol, register explicitly and run `LocalBackendContractSuite`
with a fresh fixture and effect counter. `ScriptedLocalBackend` provides an
independent offline implementation. Synthetic outcomes are not runtime proof.
A verified Docker-compatible Unix backend can share the application's Docker
workload executor. Another transport needs an executor adapter. Applications
retain their engineering state machine and evidence checks.

Configuration reference: [Lima 2.2.0 defaults](https://raw.githubusercontent.com/lima-vm/lima/v2.2.0/templates/default.yaml).
