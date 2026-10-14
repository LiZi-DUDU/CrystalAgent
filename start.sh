#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export CRYSTALAGENT_ROOT="${CRYSTALAGENT_ROOT:-$project_dir}"
cd "$project_dir"
exec python -m uvicorn backend.main:app --host "${HOST:-0.0.0.0}" --port "${PORT:-8004}"
