# ADR 0002: Establish an immutable release contract

- **Status:** Accepted
- **Date:** 2026-08-16

## Context

AgentRig declares package version `0.1.0`, but a version string and a local path
dependency do not create an immutable release. Downstream applications need to
know which source revision produced their installed artifacts and need a clear
compatibility policy. The repository also must prevent accidental distribution
of AI-agent instructions, credentials, tests, and other repository-only
material.

The package is still pre-alpha, and no evidence yet selects a package registry,
release host, signing system, or publishing automation.

## Decision

AgentRig adopts a channel-neutral release contract:

1. Versions use Semantic Versioning and tags use the exact form `vX.Y.Z`.
2. Every release binds the version and tag to a full Git commit SHA.
3. Every release contains exactly one wheel and one source distribution whose
   names and embedded metadata match `pyproject.toml`.
4. `tools/validate_release.py` proves the checkout is clean, `HEAD` and an
   annotated tag resolve to the supplied commit, validates the artifacts, and
   creates a deterministic manifest containing sizes and SHA-256 digests.
5. Released versions, tags, artifacts, and manifests are immutable and never
   reused.
6. During `0.x`, minor versions may contain documented compatibility breaks;
   patch versions remain compatible within their minor line except for an
   unavoidable, explicitly documented security or correctness fix.
7. Public APIs are the symbols exported by their owning package's `__init__.py`.
8. Publication channel, signing, and credentials remain separate decisions.

## Consequences

- A downstream project can distinguish a development checkout from a real,
  immutable release.
- Release artifacts can be verified independently of the eventual hosting
  channel.
- Repository-only and credential-like content is rejected before distribution.
- Pre-1.0 development remains possible, but compatibility changes must be
  reflected in version selection and release notes.
- Creating a tag or publishing artifacts remains an explicit release-owner
  operation; validation does not perform external side effects.
