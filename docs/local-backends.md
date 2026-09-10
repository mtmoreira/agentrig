# Replaceable local backends

The local SDK candidate `0.3.1.dev4` adds `LocalBackend` in `capabilities`,
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
added; candidate packaging is 0.3.1.dev4 and public 0.3.0 remains unchanged.

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

## Backend substitution

Implement the protocol, register explicitly and run `LocalBackendContractSuite`
with a fresh fixture and effect counter. `ScriptedLocalBackend` provides an
independent offline implementation. Synthetic outcomes are not runtime proof.
A verified Docker-compatible Unix backend can share the application's Docker
workload executor. Another transport needs an executor adapter. Applications
retain their engineering state machine and evidence checks.

Configuration reference: [Lima 2.2.0 defaults](https://raw.githubusercontent.com/lima-vm/lima/v2.2.0/templates/default.yaml).
