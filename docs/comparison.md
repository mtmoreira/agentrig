# Human review and runtime model preferences

`agentrig.evals.comparison` composes with normal `EvalCase`, `EvalRunner`, and
`EvalReport` execution. Applications choose datasets, execute their workflow for
each complete candidate model set, and export private review assets. The
framework owns the immutable comparison records and ranking arithmetic. It has
no provider registry, domain schemas, automatic model activation, or fallback.

`agentrig.selection` is a separate runtime-safe package depending only on `core`.
Production composition roots can read preferred model sets without importing
eval execution, ballots, or review infrastructure.

## Evidence and identities

A manifest declares ordered cases, ordered candidates, and zero or more unique
case/candidate cells. It needs at least one case and two candidates. Cells are
canonicalized to case/candidate order. Missing, pending, or failed cells remain
visible evidence and make the comparison ineligible to publish a preference.
Changing case versions, model sets, settings, asset hashes, costs, or outcomes
changes the manifest's SHA-256 digest. An application must finish execution and
asset export before collecting final ballots; it starts a new review identity
when evidence changes.

`ComparisonCase.from_eval_case` retains the versioned case identity.
`ComparisonCell.from_eval_result` composes with execution outcomes: any failed
execution, grader execution failure, or hard failing grade makes the cell
failed. Soft visual grades are advisory; human rankings remain the preference
signal. Costs do not participate in scoring or resolve ties.

Each review projection is an ordered collection of text, image, or artifact
references. Applications can export original inputs, reference images, exact
prompts, page images, reports, and finished books into the same collection.
References contain normalized relative POSIX paths and SHA-256 content hashes.
The core never dereferences files. A review host must confine file access to the
chosen private export directory, reject symlink escapes, verify hashes before
serving, and allow only safe presentation types. Repeated references to a path
must agree on its hash and media type. Labels and paths are excluded from repr.

Known cost has an explicit currency and finite nonnegative amount. `unknown`
requires null amount and currency. `estimated` and `observed` remain distinct;
neither token counts nor a missing price table imply zero cost.

## Rankings and preference publication

A ballot binds the exact manifest digest, case ID and version, and complete
ordered candidate roster. Its rank groups list every candidate exactly once,
best group first. Members of a group tie. Partial rankings, repeated candidates,
stale manifests, unknown cases, and incomplete case results are rejected.

A ledger accepts one ballot per case. Recording the identical ballot is
idempotent; changing an already used ballot ID or submitting a second ballot for
the same case fails. The user can edit a draft freely before the application
records its final ballot. Ballots and ledgers have private reprs. Their explicit
JSON serialization retains raw votes and must be stored privately.

For each case and candidate, a pairwise win contributes 1, a tie 1/2, and a loss
0. Divide by the number of other candidates, then average those scores equally
across the reviewed cases. Integer half-win totals decide exact aggregate ties
before float presentation. Every candidate is scored over the same cases;
there is no averaging over each candidate's available outputs.

`aggregate_rankings` returns progress scores, reviewed and total case counts,
and one reason: `incomplete_matrix`, `incomplete_rankings`, `tie`, or `complete`.
Only a full successful matrix with all cases ranked and one strictly highest
score has a winner. `summary.to_preference(manifest)` rejects all other states
and requires the exact bound manifest. Preference records retain the winner's
whole model set and provenance but omit raw votes.

## Version 1 JSON records

Every listed key is required during decoding. Constructors supply convenient
defaults, but serialized records never infer missing keys. Unknown keys,
unsupported schema versions, duplicate object keys in file storage, non-finite
numbers, and cross-application preferences are rejected.

- Model set: `application_id`, `model_set_id`, `version`, `models`, `settings`.
  `models` maps application-defined roles to nonempty model identifiers;
  `settings` maps names to JSON scalars, including quality, profile version, and
  repair limits. Both mappings are copied and frozen.
- Candidate: `candidate_id`, `version`, `model_set`, `label`.
- Asset: `asset_id`, `kind`, `label`, `path`, `sha256`, `media_type`.
- Projection: `assets`, an ordered asset array.
- Case: `case_id`, `version`, `review`.
- Cost: `status`, `amount`, `currency`.
- Cell: `case_id`, `candidate_id`, `status`, `review`, `cost`, `run_id`.
  Status is `succeeded`, `failed`, or `pending`; run ID may be null.
- Manifest: `schema_version: 1`, `comparison_id`, `version`, `application_id`,
  `cases`, `candidates`, `cells`.
- Ballot: `schema_version: 1`, `ballot_id`, `manifest_digest`, `case_id`,
  `case_version`, `candidate_ids`, `ranks` (an array of arrays).
- Ledger: `schema_version: 1`, `manifest_digest`, `ballots`.
- Summary: `manifest_digest`, `scores`, `cases_ranked`, `cases_total`,
  `winner_candidate_id` (nullable), `reason`, `scoring_version: 1`.
- Preference: `schema_version: 1`, `application_id`, `model_set`,
  `comparison_id`, `manifest_digest`, `scoring_version: 1`.

The manifest digest is computed from UTF-8 JSON with sorted keys, ASCII escaping,
no insignificant whitespace, and no non-finite numbers: Python
`json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
allow_nan=False)`. Serialized output is detached from the immutable records.

## Runtime integration

The consumer owns the scope string and storage path. Scope should identify the
application and workflow; the private path should identify the user. There is
no framework-global preference discovery. `JsonPreferenceStore(path=...)`
requires an existing caller-owned parent directory. It rejects final-component
and immediate-parent symlinks, limits files to 1 MiB, validates scope, and writes
a private 0600 temporary file followed by atomic replacement. Consumers must
also control ancestor directories and concurrent access to their store.

`select_model_set` resolves a persisted run configuration first, then an explicit
whole set, a saved preference, and finally the application default. An invalid
preference fails visibly. Availability and provider policy remain application
decisions. The consumer applies explicitly supplied per-field overrides to a new
run, validates that effective configuration, and persists it. A resumed run uses
its persisted configuration regardless of newer preferences or overrides.

See `examples/evals/model_comparison` for an offline composition and inspectable
synthetic artifacts. The UI/server integration is documented with
`agentrig.eval_review`; actual assets and votes remain application-owned.

## Local review UI

Export the frozen manifest as `comparison.json` alongside its retained assets in
a private directory. Use the directory's canonical path, with symlinks resolved
by the owner before launching the review host:

```sh
python -m agentrig.eval_review --root /canonical/private/review-directory --open
python -m agentrig.eval_review --root /canonical/private/review-directory \
  --preferences /existing/private/model-preferences.json --open
```

The optional preference file's parent must already exist. Omit `--open` to print
the local URL without opening a browser. The host serves only allowlisted safe
media from the selected directory and reverifies retained hashes. Original
inputs, reference images, exact prompts, and complete result collections remain
available while ranking each case. Give candidates the same rank to express a
tie. Unknown costs remain visibly unknown; estimates remain identified as
estimates, including when totals lack complete cost evidence.

For captions, export a plain-text asset whose ID is the image asset's ID plus
`-caption`. The viewer shows that text beneath the selected image. This optional
convention works for any illustrated result; the review host has no book schema.

Review every case, then explicitly submit the final rankings. A supplied
preference path receives only an eligible unique winner after complete ranking;
a tie or incomplete/failed matrix cannot publish one. Without a preference path,
the review records rankings without changing runtime selection. The server makes
no provider calls. Keep the review directory, private votes, and optional
preference file under the application's user-owned retention policy.
