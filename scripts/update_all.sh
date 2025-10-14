#!/usr/bin/env bash
set -euo pipefail

echo "== Update read-only submodules =="
git submodule update --remote third_party/llama.cpp third_party/voyager-sdk

echo "== Refresh race_stack fork on upstream =="
(
  cd src/race_stack
  git fetch --all
  git checkout main
  git rebase upstream/main
  git push --force-with-lease origin main || true

  # Rebase all your local feature branches atop new main
  for br in $(git for-each-ref --format='%(refname:short)' refs/heads/ | grep -E '^feat/'); do
    git checkout "$br"
    git rebase main
    git push --force-with-lease || true
  done
)

echo "== Refresh LLMxRobot fork on upstream =="
(
  cd src/LLMxRobot
  git fetch --all
  git checkout main
  git rebase upstream/main
  git push --force-with-lease origin main || true

  for br in $(git for-each-ref --format='%(refname:short)' refs/heads/ | grep -E '^feat/'); do
    git checkout "$br"
    git rebase main
    git push --force-with-lease || true
  done
)

echo "== Stage submodule bumps in superproject =="
git add third_party/llama.cpp third_party/voyager-sdk src/race_stack src/LLMxRobot
git commit -m "Update submodules to latest (read-only) and rebased forks" || echo "No changes to commit."
