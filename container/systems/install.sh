#!/usr/bin/env bash
# Native install of the per-system GPU envs: bash container/systems/install.sh <system-dir> [prefix-root]
# (progemu-prism | unimedvl). Conda layer from conda-forge only, pip layer from the hashed lock.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
sys="$1"; root="${2:-/path/to/home/myconda/envs}"
name="$(awk '/^name:/{print $2}' "$here/$sys/environment.yml")"
P="$root/$name"
# the conda on PATH (MIMIC server), else this cluster's install
if command -v conda >/dev/null 2>&1; then eval "$(conda shell.bash hook)"
else source /path/to/home/myconda/etc/profile.d/conda.sh; fi
[ -d "$P" ] || conda env create -q -p "$P" -f "$here/$sys/environment.yml"
# torch/CUDA wheels unpack ~10 GB: point TMPDIR at a large disk when /tmp is small (the development cluster: 10 GB)
export TMPDIR="${TMPDIR:-/tmp}"
"$P/bin/python" -m pip install -q --no-deps --require-hashes -r "$here/$sys/requirements.lock.txt"
"$P/bin/python" -c "import shutil, sysconfig; shutil.copy('$here/../spectra-libstdcxx.pth', sysconfig.get_paths()['purelib'])"
"$P/bin/python" -m pip check
echo "installed $P"
