# Running SpecTra on the MIMIC server

This is the first MIMIC batch: environment and self-test, the cohort census (R010), the generation smoke test
(R004) and the report extraction (R011). Later steps (resolver, E1/E2 construction, reference training, main
evaluation) land in this repository as they are written; `git pull` before each new step.

**What may leave the MIMIC server:** only the aggregate files named in "Send back" below (counts, timings,
outcome tallies, no identifiers, no text). Read each one before sending it. Everything else stays on the server:
images, reports, per-patient or per-query tables, extractions, generated images, logs.

## 0. Check the machine

```bash
ldd --version | head -1          # glibc; the envs are built for glibc >= 2.28
nvidia-smi                       # GPU model and memory, driver (>= 525.60.13 for CUDA 12; >= 575 recommended)
command -v conda apptainer podman docker
df -h .                          # ~200 GB free for envs and model weights
curl -sI https://huggingface.co | head -1    # outbound internet?
```

## 1. Code

```bash
git clone https://github.com/hq0709/spectra.git && cd spectra
mkdir -p ../third_party && cd ../third_party           # official code of three evaluated systems (R004 only)
git clone https://github.com/Masaaki-75/progemu   progemu  && git -C progemu  checkout 28cc43a885fe7a4d835bf735d5ea85f29ad3d9b2
git clone https://github.com/uni-medical/UniMedVL unimedvl && git -C unimedvl checkout f51aa0accbd02e846e5ea20b595fb39b516dbe1f
git clone https://github.com/Amarkr1/PRISM       prism    && git -C prism    checkout 8c6c6426480e64c495f5a7353ab4dfd4e02fe3d7
git clone https://github.com/virajprabhu/LANCE    lance    && git -C lance    checkout 28c39b8fc970b556729c38cd9f573432a3a5b115
cd ../spectra
export SPECTRA_THIRD_PARTY=$(realpath ../third_party)
```

## 2. Environments (conda-forge only)

| env | for | recipe |
|---|---|---|
| `spectra` | census, tests, CPU steps | `conda create -p <envs>/spectra --file container/conda-linux-64.lock`, then `pip install --no-deps --no-build-isolation -e .` |
| `spectra-gpu` | vLLM report extraction; Qwen-Image-Edit, FLUX.2 klein, Lingshu, MedGemma | `container/README.md`, section "The two envs" (conda lock + `requirements-gpu.lock.txt`) |
| `spectra-progemu-prism` | ProgEmu, PRISM (R004 only) | `bash container/systems/install.sh progemu-prism <envs>` |
| `spectra-unimedvl` | UniMedVL (R004 only) | `bash container/systems/install.sh unimedvl <envs>` |

Set `TMPDIR` to a large disk first; the CUDA wheels unpack to about 10 GB. Container routes (Apptainer, Docker,
conda-pack) are in `container/README.md`, section "MIMIC server".

## 3. Self-test

```bash
conda activate <envs>/spectra
spectra selftest          # tests needing the open MIMIC-IV demo or TAIX-Ray metadata are skipped unless
                          # SPECTRA_MIMIC_DEMO / SPECTRA_TAIX_META point at them
```

All tests must pass or skip; stop and report a failure.

## 4. Paths

```bash
cp config/mimic.yaml config/mimic.local.yaml      # git-ignored
```

Fill `cxr_jpg_dir` (MIMIC-CXR-JPG 2.1.0: `mimic-cxr-2.0.0-metadata.csv.gz`, `mimic-cxr-2.0.0-split.csv.gz`),
`cxr_reports_dir` (MIMIC-CXR 2.1.0, the unpacked `files/pXX/pXXXXXXXX/sXXXXXXXX.txt` tree) and `mimiciv_dir`
(MIMIC-IV 3.1 with `icu/` and `hosp/`). Do not change `split_salt` or `split_fractions`.

## 5. R010: cohort census (CPU, minutes)

```bash
python scripts/r010_mimic_census.py --config config/mimic.local.yaml \
       --out ../runs/R010/census.json --work ../work/mimic
```

Builds the film table (rule F1: one frontal image per study), ICU-stay membership, the patient splits (forced
test: p19 and the official validate/test patients) and the E1 funnel rows 0-4 by split. `../work/mimic/films.parquet`
is patient-level and stays; `census.json` holds counts only.

**Send back:** `../runs/R010/census.json`.

## 6. R004: one generation per system (GPU, about 1 h)

Model files (needs `HF_HOME` and, for the gated RoentGen-v2, RadEdit and MedGemma 1.5, a token of an account that
accepted their terms on Hugging Face):

```bash
python scripts/fetch_tokenizers.py
python scripts/fetch_weights.py --machine mimic
export HF_HOME=<hf cache> HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 SPECTRA_THIRD_PARTY=<third_party dir>
python scripts/r004_smoke.py --mimic-config config/mimic.local.yaml --out ../runs/R004/mimic \
       --envs <envs> --gpus 0
```

The prior is the first dev_train film with a next frontal film in the same ICU stay 1 h to 7 d later; its input is
the compiled Track-A window between the two films (this also runs the event compiler on full MIMIC-IV). Each system
runs in its own env and process; a failure is recorded and the next system runs.

**Send back:** `../runs/R004/mimic/summary.json` (per system: ok / failed, seconds per generation, peak GPU memory).

## 7. R011: report extraction (GPU, 10-40 GPU-h)

Qwen3-32B through vLLM, temperature 0, JSON-schema decoding, evidence-span and date checks, prompt frozen in
`src/spectra/references/schema.py`. Reports of all ICU films (F1) are extracted, sharded and resumable.

```bash
conda activate <envs>/spectra-gpu
huggingface-cli download Qwen/Qwen3-32B          # ~65 GB in HF_HOME
export HF_HOME=<hf cache> HF_HUB_OFFLINE=1
# 80 GB GPUs: one GPU per shard
spectra run --out ../work/extract_v4 --n-shards 64 \
    extract-reports --source mimic --config config/mimic.local.yaml --model Qwen/Qwen3-32B
# 40-48 GB GPUs: two GPUs per shard
spectra run --out ../work/extract_v4 --n-shards 64 --gpus-per-shard 2 \
    extract-reports --source mimic --config config/mimic.local.yaml --model Qwen/Qwen3-32B --tp 2
```

Re-running the same command skips finished shards. Each shard writes `extractions.*.jsonl` (patient-level, stays)
and `summary.*.json` (counts of accepted / rejected outputs by reason, rate, mean output tokens).

**Send back:** `../work/extract_v4/summary.*.json` and `../work/extract_v4/run.json`.

## Never send

MIMIC images or reports, `films.parquet`, `extractions.*.jsonl`, generated images, per-query records, logs, or any
file with subject, study, stay or dicom identifiers.
