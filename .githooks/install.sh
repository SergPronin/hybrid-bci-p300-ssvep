#!/bin/bash
# Run once after cloning: sets up graph auto-update git hooks.
# Usage: bash .githooks/install.sh

REPO_ROOT="$(git rev-parse --show-toplevel)"
HOOKS_DIR="$REPO_ROOT/.git/hooks"

echo "Installing git hooks..."

cp "$REPO_ROOT/.githooks/pre-commit"  "$HOOKS_DIR/pre-commit"
cp "$REPO_ROOT/.githooks/post-commit" "$HOOKS_DIR/post-commit"
chmod +x "$HOOKS_DIR/pre-commit" "$HOOKS_DIR/post-commit"

# Point git to .githooks so future hooks auto-apply
git config core.hooksPath .githooks

echo "✅ Hooks installed. GraphNexus + CodeGraph will auto-update on commit."
