#!/usr/bin/env bash
# One-time developer bootstrap (docs/development/07_DEVELOPMENT_
# ENVIRONMENT.md §7.9). Installs `pre-commit` and registers it as this
# clone's actual git hook (.git/hooks/pre-commit), so day-to-day
# `git commit` runs clang-format/Prettier/markdownlint automatically,
# without every developer needing to remember the individual commands.
#
# This step cannot run itself on `git clone` -- letting a clone
# auto-execute code would let any repository run arbitrary commands on
# a developer's machine just by being cloned. Run this once, by hand,
# right after cloning:
#
#   ./tools/setup-dev.sh
set -euo pipefail
cd "$(dirname "$0")/.."

if ! command -v python3 >/dev/null 2>&1; then
  echo "error: python3 not found; install Python 3 first" >&2
  exit 1
fi

python3 -m pip install --user pre-commit
python3 -m pre_commit install

echo "Tensora development hooks installed."
echo "git commit now runs clang-format (C/C++) and Prettier + markdownlint (Markdown) automatically."
echo "Skip a hook run for one commit with: git commit --no-verify (not recommended; CI re-checks anyway)."
