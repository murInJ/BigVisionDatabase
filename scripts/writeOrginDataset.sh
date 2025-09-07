#!/usr/bin/env bash
set -euo pipefail

# --------------------------------------
# writeOriginDataset.sh
# Convenience runner for OriginDataset/writer.py (DuckDB-only)
# --------------------------------------
# Options:
#   -r, --force           Force rewrite (dangerous; deletes existing dataset data)
#   -n, --dry-run         Dry run (no writes; prints actions)
#   -w, --workers  N      Max concurrent image writers (default: 8)
#   -d, --db-root  PATH   Override database_root from Config.setting
#   -y, --yes             Skip confirmation prompts when forcing rewrite
#   -h, --help            Show help
#
# Env vars alternative: DB_ROOT, MAX_WORKERS, FORCE_REWRITE=1, DRY_RUN=1
#
# This script sets PYTHONPATH to the repo root so that `OriginDataset` is importable.

FORCE_REWRITE=${FORCE_REWRITE:-0}
DRY_RUN=${DRY_RUN:-0}
MAX_WORKERS=${MAX_WORKERS:-8}
DB_ROOT=${DB_ROOT:-}
ASSUME_YES=0

print_help() {
  cat <<'USAGE'
writeOriginDataset.sh - run OriginDataset/writer.py to write datasets discovered by registry

Options:
  -r, --force           Force rewrite (dangerous; deletes existing dataset data)
  -n, --dry-run         Dry run (no writes; prints actions)
  -w, --workers  N      Max concurrent image writers (default: 8)
  -d, --db-root  PATH   Override database_root from Config.setting
  -y, --yes             Skip confirmation prompts when forcing rewrite
  -h, --help            Show this help
USAGE
}

# Parse args
while [[ ${1:-} != "" ]]; do
  case "$1" in
    -r|--force)   FORCE_REWRITE=1; shift;;
    -n|--dry-run) DRY_RUN=1; shift;;
    -w|--workers) MAX_WORKERS=${2:?--workers requires a number}; shift 2;;
    -d|--db-root) DB_ROOT=${2:?--db-root requires a path}; shift 2;;
    -y|--yes)     ASSUME_YES=1; shift;;
    -h|--help)    print_help; exit 0;;
    --) shift; break;;
    *) echo "Unknown option: $1" >&2; print_help; exit 2;;
  esac
done

# Compute repo root as the directory containing this script
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT="${SCRIPT_DIR}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

# Preflight: python3 present
if ! command -v python3 >/dev/null 2>&1; then
  echo "Error: python3 not found in PATH." >&2
  exit 127
fi

# Preflight: duckdb available
if ! python3 - <<'PY' >/dev/null 2>&1; then
import sys
try:
    import duckdb  # noqa: F401
except Exception:
    sys.exit(1)
else:
    sys.exit(0)
PY
then
  echo "Error: DuckDB is required. Install with: python3 -m pip install duckdb" >&2
  exit 1
fi

# Force rewrite confirmation
if [[ "$FORCE_REWRITE" == "1" && "$ASSUME_YES" != "1" && "$DRY_RUN" != "1" ]]; then
  echo "
[WARNING] You are about to FORCE rewrite datasets discovered by registry."
  echo "This will delete existing metadata in DuckDB and replace the images directories."
  read -r -p "Type EXACTLY 'OVERWRITE' to proceed: " ANSWER
  if [[ "$ANSWER" != "OVERWRITE" ]]; then
    echo "Aborted."
    exit 1
  fi
fi

# Run writer
exec python3 - <<PY
import os
from Config.setting import GetDatabaseConfig
from OriginDataset.writer import write

cfg = GetDatabaseConfig()
# Override database_root if provided
_db_root = os.environ.get('DB_ROOT')
if _db_root:
    cfg['database_root'] = _db_root

force_rewrite = os.environ.get('FORCE_REWRITE','0') == '1'
dry_run = os.environ.get('DRY_RUN','0') == '1'
max_workers = int(os.environ.get('MAX_WORKERS','8'))

print(f"Using database_root: {cfg['database_root']}")
print(f"force_rewrite={force_rewrite}, dry_run={dry_run}, max_workers={max_workers}")

write(cfg, force_rewrite=force_rewrite, dry_run=dry_run, max_workers=max_workers)
PY
