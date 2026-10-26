# SpecTra runtime: one locked env, three ways to run it

The same conda-forge environment runs on the the development cluster Slurm cluster (EL8,
glibc 2.28) and on the MIMIC server (OS unknown, no Slurm assumed). It is
defined once, locked once, and installed only from the locks, so a native
env, a container image and a conda-pack tarball hold the same packages.

| File | Purpose |
|---|---|
| `environment-core.yml` | Gate-0 env `spectra`: Python 3.11, CPU only, exact top-level pins |
| `environment-gpu.yml` | `spectra-gpu`: core + CUDA 12.9 PyTorch/vLLM stack, exact pins |
| `virtual-packages.yml` | what the locks assume about the host: linux-64, glibc 2.28, kernel 4.18, no `__cuda` |
| `conda-lock.yml`, `conda-linux-64.lock` | core lock (conda-lock unified / explicit), 132 packages with hashes |
| `conda-lock-gpu.yml`, `conda-gpu-linux-64.lock` | GPU env, conda layer (core + llguidance), 136 packages |
| `requirements-gpu.lock.txt` | GPU env, pip layer, 185 packages, sha256-pinned |
| `spectra-libstdcxx.pth` | GPU env only: preloads the env's libstdc++ (see "libstdc++ on EL8") |
| `relock.sh` | regenerates all locks (`core`, `gpu`, `all`) |
| `Containerfile` | Docker/Podman image, `--build-arg FLAVOR=core\|gpu` |
| `spectra.def` | Apptainer image, `--build-arg FLAVOR=core\|gpu` |
| `build_condapack.sh` | relocatable env tarball (fallback when no container runtime) |
| `dist/` (git-ignored) | built artifacts: `spectra-core.sif`, `spectra-core-env.tar.gz` (+ `.sha256`) |

## The two envs

**core** (conda-forge, strict, `nodefaults`): python 3.11.17, numpy 2.3.5,
pandas 3.0.6, pyarrow 25.0.0, duckdb 1.5.6, scipy 1.17.1, pyyaml 6.0.3,
pytest 9.1.1, scikit-learn 1.9.1, statsmodels 0.15.0, tqdm 4.70.1,
jsonschema 4.26.0, plus pip 26.2.1 / setuptools 84.2.0 / wheel 0.48.0 so that
`pip install --no-build-isolation -e .` works offline.

**gpu** = core + `llguidance 1.7.6` (conda-forge) + pip:

| package | pin | from |
|---|---|---|
| torch | 2.13.0+cu129 | download.pytorch.org/whl/cu129 |
| torchvision | 0.28.0+cu129 | download.pytorch.org/whl/cu129 |
| torchaudio | 2.11.0+cu129 | download.pytorch.org/whl/cu129 (hard requirement of vllm) |
| vllm | 0.32.0+cu129 | GitHub release asset, sha256 `9918be80…bb50b` |
| transformers | 5.19.0 | PyPI (vllm needs >=5.16.1,<5.20.0) |
| diffusers | 0.41.0 | PyPI |
| accelerate | 1.15.0 | PyPI |
| safetensors | 0.8.0 | PyPI |
| pillow | 12.3.0 | PyPI (13.0.0 was 4 days old at lock time) |
| huggingface_hub | 1.33.0 | PyPI (diffusers 0.41 needs <2.0) |
| sentencepiece | 0.2.2 | PyPI |

How the torch/vLLM pairing was established (not guessed):
- vLLM v0.20.0 release notes moved the PyPI wheel to CUDA 13.0 and say to
  use the cu129 build on CUDA 12.9. The v0.32.0 release ships
  `vllm-0.32.0+cu129-cp38-abi3-manylinux_2_28_x86_64.whl`.
- That wheel's own METADATA pins `torch==2.13.0`, `torchvision==0.28.0` and
  `torchaudio==2.11.0`, the same as `requirements/cuda.txt` at tag v0.32.0.
  It also switches `humming-kernels` from the `[cu13]` extra in
  `requirements/cuda.txt` to `[cu12]`.
- download.pytorch.org/whl/cu129 serves all three torch packages for cp311.
- `uv pip compile` resolved the whole tree (185 packages) for
  manylinux_2_28 from wheels only. The result is CUDA 12.9 throughout:
  `cuda-toolkit 12.9.1`, `cuda-bindings 12.9.10`, `nvidia-*-cu12`. The one
  exception is `nvidia-cutlass-dsl-libs-cu13`, a hard dependency of
  `nvidia-cutlass-dsl 4.8.0` used only by the Hopper/Blackwell kernels.

The vLLM tree forces two conda-side choices:
- **numpy 2.3.5 in both envs.** mistral-common needs numpy<2.4 on
  Python < 3.13. Pinning core the same way keeps Gate-0 and the GPU gates on
  one numpy build.
- **llguidance from conda-forge.** vllm needs llguidance >=1.7,<1.8. PyPI's
  CPython wheels for 1.7.x are manylinux_2_31 and do not install on EL8
  (glibc 2.28); the conda-forge build is abi3 and needs glibc >= 2.17.

GPU host requirements: an NVIDIA driver >= 525.60.13 (CUDA 12 minor-version
compatibility), >= 575 recommended. the development cluster has 580.159.04. CUDA comes from the
pip wheels, so no CUDA toolkit is needed on the host or in the image.

### libstdc++ on EL8

On the development cluster, `import torch` before numpy fails in a conda env:

```
ImportError: /lib64/libstdc++.so.6: version `GLIBCXX_3.4.29' not found (required by .../numpy/_core/_multiarray_umath...so)
```

The torch wheel's libraries carry their own RUNPATH, so their
`libstdc++.so.6` resolves to EL8's system copy (GCC 8, up to GLIBCXX_3.4.25).
conda-forge numpy needs the env's copy (up to GLIBCXX_3.4.36), and the
process can hold only one. `spectra-libstdcxx.pth` loads the env's copy at
interpreter start. It works with or without activation, under `python -I`,
and in vLLM's worker processes, and it leaves system tools alone (an
`LD_LIBRARY_PATH` hook would affect them too). The image and conda-pack
builds install it. To create the GPU env natively:

```bash
P=/path/to/home/myconda/envs/spectra-gpu
conda create -p $P --file container/conda-gpu-linux-64.lock
export TMPDIR=/path/to/home/tmp   # pip unpacks ~10 GB of torch/CUDA wheels; the development cluster's /tmp is 10 GB
$P/bin/python -m pip install --no-deps --require-hashes -r container/requirements-gpu.lock.txt
$P/bin/python -c 'import shutil, sysconfig; shutil.copy("container/spectra-libstdcxx.pth", sysconfig.get_paths()["purelib"])'
```

The env is about 17 GB installed. On the development cluster that download took 25 minutes.

FlashInfer's precompiled kernels are not in the lock because they are built
per GPU architecture. Without them, vLLM downloads or compiles kernels at
startup. On a node that sees the target GPU and has internet, run
`vllm download-kernels` once in the GPU env.

## Per-system envs (official code of ProgEmu, PRISM, UniMedVL)

Three panel systems run through their released code, which pins libraries the GPU env cannot hold
(transformers 5 removed internals ProgEmu imports; PRISM pins diffusers 0.24; UniMedVL imports flash-attn at
module level). Each gets its own env, locked the same way (conda-forge Python + hashed `uv pip compile` lock,
torch from PyPI, whose Linux wheels are the CUDA 12 builds):

| env | systems | key pins | lock |
|---|---|---|---|
| `spectra-progemu-prism` | ProgEmu, PRISM | python 3.11, torch 2.4.1 (CUDA 12.1), transformers 4.45.2, diffusers 0.24.0, huggingface_hub 0.25.2, flash-attn 2.6.3 | `systems/progemu-prism/` |
| `spectra-unimedvl` | UniMedVL | python 3.10, torch 2.5.1 (CUDA 12.4), transformers 4.49.0, flash-attn 2.7.4.post1 | `systems/unimedvl/` |

```bash
TMPDIR=/path/to/home/tmp bash container/systems/install.sh progemu-prism   # or: unimedvl
```

`systems.yaml` names each system's env (`env`) and code commit (`code`); `scripts/r004_smoke.py` runs every
system in its own env. Both envs were installed on the development node on 2026-10-21; per-system R004 outcomes are in
`refine-logs/EXPERIMENT_TRACKER.md`.

## Smoke test

`spectra selftest` runs the Gate-0 suite and is the check on every machine.
Where the CLI is not installed, use `python -m pytest -q /opt/spectra/tests`
inside an image, or `python -m pytest -q tests` in a checkout. Both
`selftest` and the config loader find `tests/` and `config/` relative to the
source tree (`SPECTRA_CONFIG_DIR` overrides the config path), so spectra runs
from a checkout or an editable install, not from a built wheel.

## the development cluster (this cluster)

House rules still apply: `ulimit -u 65536;` first. Anything that puts a model
on a GPU goes through `sbatch`. Measured peaks for this account: about 408
threads during the podman build and 376 during the apptainer build.

**Native dev env** (Gate-0):

```bash
source /path/to/home/myconda/etc/profile.d/conda.sh
conda activate spectra        # /path/to/home/myconda/envs/spectra
pip install --no-deps --no-build-isolation -e /path/to/spectra
spectra selftest
```

The editable install writes `src/spectra.egg-info`. To run without installing:
`PYTHONPATH=src python -m spectra.cli selftest`, from the spectra root.

To rebuild it from the lock: `conda create -p .../envs/spectra --file container/conda-linux-64.lock`.

**Apptainer.** There is no system Apptainer on the login node. The
`spectra-tools` env carries apptainer 1.5.4 (conda-forge, unprivileged).
Without root it builds in a root-mapped user namespace via `--fakeroot`:

```bash
cd /path/to/spectra
T=/path/to/home/myconda/envs/spectra-tools
APPTAINER_CACHEDIR=$PWD/container/.build/apptainer-cache \
  $T/bin/apptainer build --fakeroot container/dist/spectra-core.sif container/spectra.def
$T/bin/apptainer run container/dist/spectra-core.sif selftest
# run the current checkout with the locked env (read-only, so no pytest cache):
$T/bin/apptainer run --bind $PWD:/opt/spectra:ro --env PYTEST_ADDOPTS="-p no:cacheprovider" \
  container/dist/spectra-core.sif selftest
# GPU gates, on a compute node via sbatch:
#   apptainer build --fakeroot --build-arg FLAVOR=gpu container/dist/spectra-gpu.sif container/spectra.def
#   sbatch -p gen-a100.p --gres=gpu:a100:1 ... --wrap 'apptainer exec --nv spectra-gpu.sif ...'
```

Whether the compute nodes have Apptainer is not checked. The conda-forge
binary from `spectra-tools` works anywhere `network storage` is mounted.

**Podman (rootless).** The account has no `/etc/subuid` range, so podman
maps one UID. A plain build then fails while unpacking the base image:

```
Error: ... potentially insufficient UIDs or GIDs available in user namespace
(requested 0:42 for /etc/gshadow) ... lchown /etc/gshadow: invalid argument
```

containers-storage has an option for this case that squashes all layer
owners to the one mapped UID. Pull the base once with it; after that the
plain build works (tested with `--no-cache`):

```bash
podman --storage-opt overlay.ignore_chown_errors=true pull \
  docker.io/mambaorg/micromamba:2.9.0@sha256:e0a99b0f17a759e14c2f967dc0ca2d3a3c1ca3c62955f4d20bba770eaaf0184d
podman build -f container/Containerfile -t spectra-core:0 /path/to/spectra
podman run --rm spectra-core:0 selftest                  # source baked in at build time
podman run --rm -v "$PWD":/opt/spectra:ro -e PYTEST_ADDOPTS="-p no:cacheprovider" \
  spectra-core:0 selftest                                # current checkout, from the spectra root
podman run --rm --entrypoint python spectra-core:0 -m pytest -q /opt/spectra/tests
```

Images live in `/path/to/podman/storage`, which is local to this node.
Because of the squash, every file in the image is owned by root. The image
also runs as root, which inside rootless podman is the invoking user.

## MIMIC server

Pick the first route the server supports. Never put MIMIC data under
`spectra/` before a build: the images copy only `pyproject.toml`,
`README.md`, `src/`, `config/` and `tests/`. Mount data read-only at run time
(`--bind /data/mimic:/data/mimic:ro` or `-v /data/mimic:/data/mimic:ro`).

1. **Apptainer / SingularityCE.** Copy `dist/spectra-core.sif`, then run
   `apptainer run spectra-core.sif selftest`. A SIF is a single file, needs no
   daemon and no internet, and runs as the calling user. For GPUs, add `--nv`.
2. **Docker / Podman.** Either build there from this directory (needs
   internet to conda-forge, plus PyPI for `FLAVOR=gpu`), or move the image
   built here:
   ```bash
   podman save --format docker-archive -o spectra-core-0.tar localhost/spectra-core:0   # here
   docker load -i spectra-core-0.tar                                                    # there
   docker run --rm --user "$(id -u):$(id -g)" spectra-core:0 selftest
   ```
   For GPUs, use `docker run --gpus all` (nvidia-container-toolkit) or
   `podman run --device nvidia.com/gpu=all` (CDI). The same tar also converts
   to a SIF: `apptainer build spectra-core.sif docker-archive://spectra-core-0.tar`.
3. **conda-pack**, when no container runtime exists. The host needs
   linux-64 and glibc >= 2.28 (EL8+, Ubuntu 20.04+, Debian 11+).
   ```bash
   # here: build, and test the round trip into a scratch dir (keep it off the development cluster's small /tmp)
   CONDA=/path/to/home/myconda/bin/conda \
     container/build_condapack.sh --flavor core --test-unpack container/.build/unpack-test
   # there: copy dist/spectra-core-env.tar.gz (+ .sha256) and a spectra checkout
   sha256sum -c spectra-core-env.tar.gz.sha256
   mkdir -p ~/envs/spectra && tar -xzf spectra-core-env.tar.gz -C ~/envs/spectra
   ~/envs/spectra/bin/conda-unpack
   source ~/envs/spectra/bin/activate
   pip install --no-deps --no-build-isolation -e /path/to/spectra    # offline
   spectra selftest
   ```
   `--flavor gpu` packs the GPU env the same way. It is about 17 GB
   installed, and that tarball has not been built.

## Changing a pin

Edit the yml, then run `container/relock.sh core|gpu`. That uses conda-lock,
uv and conda from the `spectra-tools` env and base install, and keeps
thread pools small. After relocking, rebuild the dev env, the images and the
tarball from the new locks.

Always install from the locks, never by solving a yml. Re-solving the
unchanged `environment-core.yml` one hour after it was locked (same content
hash) already gave 134 packages instead of 132, e.g. libcurl
8.21.0 → 8.23.0, new libsqlite/libxml2 builds, and icu and libpsl added.

The GPU lock comes in two halves. conda-lock 4.0.3 does resolve the pip layer,
but then fails to pick NVIDIA's wheels: its manylinux tag list
(`1, 2010, 2014, _2_17, _2_18, _2_24, _2_28`) skips `_2_27`, the tag on the
CUDA 12.9 libraries (`Unable to find installation candidates for
nvidia-cublas-cu12 (12.9.1.4)`). So `relock.sh gpu` locks the conda layer
with conda-lock and installs it into a throwaway prefix. It then runs
`uv pip compile --generate-hashes --torch-backend cu129` against exactly
those installed versions, and drops everything the conda layer already
provides.

## Tested here vs not

Tested on the development node (RHEL 8.10, glibc 2.28, 2026-10-20/21). The test suite
changed while this was being built, so each result names the source it ran on.

- **Core lock to dev env.** Installed set equals `conda-linux-64.lock`
  (132/132, URL + md5). `pip check` is clean, and imports plus a functional
  smoke test (parquet round trip, duckdb, OLS, sklearn) pass.
- **Podman core image** (`localhost/spectra-core:0`, 1.22 GB). Built with the
  workaround above, and again with a plain `--no-cache` build after the
  one-time pull. The image's conda set equals the lock (132/132).
  `spectra --help` works. `spectra selftest` with the 23:35 source mounted
  read-only: 29 passed, 9 skipped.
- **Apptainer core SIF** (`dist/spectra-core.sif`, 370 MB). Built
  unprivileged from `spectra.def` with conda-forge apptainer 1.5.4
  `--fakeroot`, and `%test` passed. The SIF's conda set equals the lock, and
  the runscript works.
- **conda-pack round trip.** Ran `build_condapack.sh --test-unpack`: packed
  `dist/spectra-core-env.tar.gz` (289 MB) and unpacked it into a scratch dir.
  Then ran `conda-unpack` (no file left pointing at the build prefix),
  deleted the build prefix, and ran
  `python -c "import numpy, pandas, pyarrow, duckdb, scipy, yaml"` and
  `pip check`. `spectra selftest` on a mirrored copy of the 00:04 tree:
  75 passed, 8 skipped.
- **Parity.** Source edits at 23:43–23:45 caused 3 failures in
  `tests/test_components.py`, identical in the native dev env and in the
  SIF. Later edits fixed them.
- **Read-only mounts.** The newer `test_wheel_carries_the_resources_and_runs_relocated`
  runs `pip wheel` on the tree it lives in, which writes `build/` and
  `src/spectra.egg-info`. It therefore fails whenever the source is mounted
  read-only (the podman and SIF runs above), and dirties a writable checkout.
  That is why `build_condapack.sh` tests a copy.
- **GPU lock.** It resolves for manylinux_2_28 from wheels only. It was
  installed natively into a throwaway prefix from `conda-gpu-linux-64.lock`
  plus `pip install --no-deps --require-hashes -r requirements-gpu.lock.txt`,
  and `pip check` is clean. With `spectra-libstdcxx.pth` and
  `CUDA_VISIBLE_DEVICES=""` (no GPU touched), these all import under
  `python -I` with torch imported first: torch 2.13.0+cu129 (CUDA 12.9,
  cuDNN 9.20), torchvision 0.28.0+cu129, torchaudio 2.11.0+cu129,
  transformers 5.19.0, diffusers 0.41.0, accelerate 1.15.0,
  safetensors 0.8.0, pillow 12.3.0, huggingface_hub 1.33.0,
  sentencepiece 0.2.2, vllm 0.32.0 and numpy 2.3.5. Without the `.pth`,
  `import torch` fails as shown above.

Not tested:

- Anything on a GPU: no CUDA initialisation, no model load, no vLLM serving,
  no `--nv` / `--gpus` / CDI passthrough.
- `FLAVOR=gpu` image builds and the GPU conda-pack tarball.
- Docker itself (only podman/buildah here). The Containerfile uses only
  standard syntax.
- The MIMIC server, and Apptainer on the development cluster compute nodes.
