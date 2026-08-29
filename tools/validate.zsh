#!/bin/zsh

# Durable macOS wrapper for the complete offline AgentRig candidate gate.
# It never enables live tests and is safe to invoke from an interactive shell.

set +e
set +u
set +o pipefail

validation_log="$(mktemp -t agentrig-validation.XXXXXX)"

(
  set -euo pipefail

  if (( $# != 0 )); then
    print -u2 'usage: zsh tools/validate.zsh'
    false
  fi

  script_directory="${0:A:h}"
  repository_root="${script_directory:h}"
  candidate_directory="$(mktemp -d -t agentrig-candidate.XXXXXX)"
  trap 'rm -rf -- "$candidate_directory"' EXIT
  cd "$repository_root"

  print 'CHECKPOINT repository-root'
  [[ "$(git rev-parse --show-toplevel)" == "$repository_root" ]]

  print 'CHECKPOINT staging-area-empty'
  [[ -z "$(git diff --cached --name-only)" ]]

  print 'CHECKPOINT capture-worktree'
  initial_worktree="$(git status --porcelain=v1 --untracked-files=all)"

  print 'CHECKPOINT uv-version'
  [[ "$(uv --version)" == "uv 0.12.3 "* ]]

  print 'CHECKPOINT lock-current'
  uv lock --check

  print 'CHECKPOINT locked-environment'
  uv sync --frozen --all-extras

  print 'CHECKPOINT python-dependency-bridge'
  uv run python tools/generate_buck_python_deps.py --check

  print 'CHECKPOINT agent-context'
  uv run python tools/validate_agent_context.py

  print 'CHECKPOINT strict-typecheck'
  uv run python tools/typecheck.py

  print 'CHECKPOINT unit-tests'
  uv run python -m unittest discover -s tests/unit -t . -v

  print 'CHECKPOINT full-offline-buck'
  ./buck2 test //... --exclude live --always-exclude

  print 'CHECKPOINT package-build'
  uv build --out-dir "$candidate_directory"
  wheel="$candidate_directory/agentrig-0.3.0-py3-none-any.whl"
  [[ -f "$wheel" ]]

  print 'CHECKPOINT isolated-base'
  uv run --isolated --no-project --with "$wheel" \
    python -c 'import agentrig; assert agentrig.__all__ == ()'

  for extra in codex ollama openai; do
    print "CHECKPOINT isolated-${extra}-extra"
    uv run --isolated --no-project --with "${wheel}[${extra}]" \
      python -c 'import agentrig; assert agentrig.__all__ == ()'
  done

  print 'CHECKPOINT whitespace-and-diff'
  git diff --check

  print 'CHECKPOINT worktree-preserved'
  [[ "$(git status --porcelain=v1 --untracked-files=all)" == "$initial_worktree" ]]

  print 'CHECKPOINT staging-area-still-empty'
  [[ -z "$(git diff --cached --name-only)" ]]

  print 'AgentRig complete offline candidate validation passed'
  print 'No live test or provider service was run'
) 2>&1 | tee "$validation_log"
validation_status=${pipestatus[1]}

print
print "AGENTRIG_VALIDATION_STATUS=${validation_status}" | tee -a "$validation_log"
if (( $+commands[pbcopy] )); then
  pbcopy < "$validation_log"
  print 'Validation report copied to the clipboard'
else
  print 'pbcopy is unavailable; copy the validation report from this terminal'
fi
rm -f "$validation_log"

(( validation_status == 0 ))
