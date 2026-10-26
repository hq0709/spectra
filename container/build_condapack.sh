#!/usr/bin/env bash
# Build a relocatable SpecTra env tarball with conda-pack: the fallback for a
# machine with no usable container runtime. The env is created from the lock
# files in this directory (no solve), packed, and can be unpacked anywhere on
# a linux-64 host with glibc >= 2.28.
#
# Usage (from anywhere):
#   container/build_condapack.sh [--flavor core|gpu] [--out DIR] [--work DIR]
#                                [--test-unpack DIR]
#
#   --flavor core        Gate-0 CPU env (default)  -> spectra-core-env.tar.gz
#   --flavor gpu         core + CUDA 12.9 stack    -> spectra-gpu-env.tar.gz
#   --out DIR            where the tarball goes (default: container/dist)
#   --work DIR           scratch for the build prefix (default: container/.build/pack.XXXXXX)
#   --test-unpack DIR    after packing, run the target-side steps here: unpack
#                        into DIR, conda-unpack, delete the build prefix,
#                        import smoke test, then `spectra selftest` from the
#                        unpacked env on a scratch copy of this source tree
#
# The tarball holds the env only. spectra itself runs from a source checkout
# (it finds config/ and tests/ relative to the source tree), so on the target:
#   mkdir -p ~/envs/spectra && tar -xzf spectra-core-env.tar.gz -C ~/envs/spectra
#   ~/envs/spectra/bin/conda-unpack        # once; rewrites the build prefix
#   source ~/envs/spectra/bin/activate
#   pip install --no-deps --no-build-isolation -e /path/to/spectra   # offline
#   spectra selftest
#
# Tools: CONDA (conda or micromamba binary) and CONDA_PACK (conda-pack
# binary). Defaults: whatever is on PATH, falling back to the the development cluster install.

set -euo pipefail
ulimit -u 65536 2>/dev/null || true   # the development cluster per-user thread cap; no-op elsewhere

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PKG_ROOT="$(dirname "$HERE")"
FLAVOR=core
OUT="$HERE/dist"
WORK=""
TEST_UNPACK=""

while [ $# -gt 0 ]; do
  case "$1" in
    --flavor) FLAVOR="$2"; shift 2 ;;
    --out) OUT="$2"; shift 2 ;;
    --work) WORK="$2"; shift 2 ;;
    --test-unpack) TEST_UNPACK="$2"; shift 2 ;;
    -h|--help) sed -n '2,30p' "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

# Absolute paths from here on: the prefix checks below compare path strings.
absdir() { mkdir -p "$1" && (cd "$1" && pwd); }
OUT="$(absdir "$OUT")"
[ -z "$WORK" ] || WORK="$(absdir "$WORK")"
if [ -n "$TEST_UNPACK" ]; then
  TEST_UNPACK="$(absdir "$(dirname "$TEST_UNPACK")")/$(basename "$TEST_UNPACK")"
fi

case "$FLAVOR" in
  core) CONDA_LOCK="$HERE/conda-linux-64.lock"; PIP_LOCK="" ;;
  gpu)  CONDA_LOCK="$HERE/conda-gpu-linux-64.lock"; PIP_LOCK="$HERE/requirements-gpu.lock.txt" ;;
  *) echo "--flavor must be core or gpu" >&2; exit 2 ;;
esac

ROHPC_CONDA=/path/to/home/myconda/bin/conda
ROHPC_PACK=/path/to/home/myconda/envs/spectra-tools/bin/conda-pack
CONDA="${CONDA:-$(command -v micromamba || command -v conda || echo "$ROHPC_CONDA")}"
CONDA_PACK="${CONDA_PACK:-$(command -v conda-pack || echo "$ROHPC_PACK")}"
[ -x "$CONDA" ] || { echo "no conda/micromamba found; set CONDA=" >&2; exit 1; }
[ -x "$CONDA_PACK" ] || { echo "no conda-pack found; set CONDA_PACK=" >&2; exit 1; }

# Keep BLAS/OpenMP pools small while building and testing; on the development cluster the
# per-user thread cap is shared with every other process of the account.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1 PYTHONNOUSERSITE=1

# Default scratch sits next to this script, not in /tmp: the development cluster's /tmp is 10 GB,
# and on the conda package cache's filesystem the env is hardlinked, not copied.
if [ -z "$WORK" ]; then mkdir -p "$HERE/.build"; WORK="$(mktemp -d "$HERE/.build/pack.XXXXXX")"; fi
mkdir -p "$WORK" "$OUT"
PREFIX="$WORK/prefix-$FLAVOR"
TARBALL="$OUT/spectra-$FLAVOR-env.tar.gz"
[ ! -e "$PREFIX" ] || { echo "build prefix exists, refusing to reuse: $PREFIX" >&2; exit 1; }

echo "== create $PREFIX from $(basename "$CONDA_LOCK") with $CONDA"
"$CONDA" create -y -p "$PREFIX" --file "$CONDA_LOCK"
PY="$PREFIX/bin/python"

if [ -n "$PIP_LOCK" ]; then
  echo "== pip layer from $(basename "$PIP_LOCK")"
  "$PY" -m pip install --no-deps --require-hashes -r "$PIP_LOCK"
  # Preload the env's libstdc++ so torch cannot pull in EL8's older system
  # copy first (README "libstdc++ on EL8"). conda-pack packs unmanaged files.
  "$PY" -c 'import shutil, sys, sysconfig; shutil.copy(sys.argv[1], sysconfig.get_paths()["purelib"])' \
      "$HERE/spectra-libstdcxx.pth"
fi

"$PY" -m pip check
"$PY" -I -c "import numpy, pandas, pyarrow, duckdb, scipy, yaml"

echo "== pack -> $TARBALL"
rm -f "$TARBALL" "$TARBALL.sha256"
"$CONDA_PACK" -p "$PREFIX" -o "$TARBALL" --n-threads 4
( cd "$OUT" && sha256sum "$(basename "$TARBALL")" > "$(basename "$TARBALL").sha256" )
ls -l "$TARBALL"; cat "$TARBALL.sha256"

if [ -n "$TEST_UNPACK" ]; then
  echo "== round trip: unpack into $TEST_UNPACK"
  [ ! -e "$TEST_UNPACK" ] || { echo "refusing to unpack into existing $TEST_UNPACK" >&2; exit 1; }
  mkdir -p "$TEST_UNPACK"
  tar -xzf "$TARBALL" -C "$TEST_UNPACK"
  "$TEST_UNPACK/bin/conda-unpack"
  # Nothing may still point at the build prefix, except conda-unpack itself
  # (it stores the old prefix to replace) and conda's command log.
  left="$(grep -rIl --exclude-dir=__pycache__ -F "$PREFIX" "$TEST_UNPACK" \
          | grep -vxF -e "$TEST_UNPACK/bin/conda-unpack" -e "$TEST_UNPACK/conda-meta/history" || true)"
  if [ -n "$left" ]; then echo "build prefix still referenced in:" >&2; echo "$left" | head >&2; exit 1; fi
  # Remove the build prefix so nothing below can silently load from it.
  rm -rf "$PREFIX"
  MIRROR=""
  if [ -f "$PKG_ROOT/pyproject.toml" ]; then
    # The suite runs on a copy, never on the real tree: one test builds a
    # wheel from its own tree, which writes build/ and src/*.egg-info. The
    # copy sits in a mirror of the parent directory (siblings symlinked),
    # because tests also read sibling paths such as ../refine-logs.
    MIRROR="$WORK/mirror"
    PARENT="$(dirname "$PKG_ROOT")"; NAME="$(basename "$PKG_ROOT")"
    rm -rf "$MIRROR"; mkdir -p "$MIRROR/$NAME"
    for s in "$PARENT"/* "$PARENT"/.[!.]*; do
      if [ -e "$s" ] && [ "$(basename "$s")" != "$NAME" ]; then ln -s "$s" "$MIRROR/"; fi
    done
    for p in pyproject.toml README.md src config tests; do
      [ ! -e "$PKG_ROOT/$p" ] || cp -r "$PKG_ROOT/$p" "$MIRROR/$NAME/"
    done
    find "$MIRROR/$NAME" \( -name '__pycache__' -o -name '*.egg-info' \) -prune -exec rm -rf {} +
  fi
  (
    set +u
    # shellcheck disable=SC1091
    source "$TEST_UNPACK/bin/activate"
    set -u
    cd "$TEST_UNPACK"
    python -I -c "import numpy, pandas, pyarrow, duckdb, scipy, yaml; print('unpacked env imports OK: numpy', numpy.__version__, 'pandas', pandas.__version__, 'duckdb', duckdb.__version__)"
    python -I -c "import sys; assert sys.prefix == '$TEST_UNPACK', sys.prefix; print('sys.prefix =', sys.prefix)"
    python -I -m pip check
    if [ -n "$MIRROR" ]; then
      PYTHONPATH="$MIRROR/$NAME/src" PYTHONDONTWRITEBYTECODE=1 \
        PYTEST_ADDOPTS="-p no:cacheprovider" python -m spectra.cli selftest
      echo "spectra selftest from the unpacked env: OK"
    fi
  )
fi
echo "done: $TARBALL"
