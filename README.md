# spectra

Constructor and scorer for **SpecTra**, trajectory (E1) and specification (E2) tests for chest-radiograph
world models. **Running on the MIMIC server: start with [`MIMIC_SERVER.md`](MIMIC_SERVER.md).**

One repository and one container run unchanged on the MIMIC server and on the Slurm cluster. Every step is
`spectra <step> --config config/<source>.yaml`. MIMIC-derived patient-level artifacts never leave the MIMIC
server; `spectra export` is the only outbound path and refuses anything that is not an aggregate table.

## Layout

| Module | Role |
|---|---|
| `cer.py` | canonical event records from MIMIC-IV ICU tables: allowlist/denylist, occurrence cutoff (availability as sensitivity), interval rows → state + transitions, candidate removals |
| `e2.py` | device state machine g_d(s0, M), background/bundle partition, edit classes and eligibility, primary cohort (7a), secondary allocation (7b), cross-source protocol, JDA / FA / EA |
| `e1.py` | query sets grouped by prior, admitted contrasts, prior sampling, within-prior concordance, conventional AUROC, absolute error |
| `serialize.py` | one compiler for every interface (typed tokens, event text, verbalization, 77-token compact form), capacity rule, token-budgeted background |
| `roundtrip.py` | mandatory round-trip and background-equality check per system tokenizer |
| `controls.py` | copy-prior, query-blind, bundle-blind, record-volume, oracle, population-rate; exact-property assertions |
| `references/` | extraction schema + evidence-span check, resolver R1–R5, device-state tiers, point-estimate validation with binomial SDs |
| `ppi.py` | two-stage audit sampling, PPI / PPI++ estimator, stratified patient bootstrap |
| `decision.py` | Holm decision table H1–H9 per reader-budget rung, divergence and cross-source claims |
| `export_aggregate.py` | aggregate-only export with identifier and free-text checks |
| `adapters/` | source contract; MIMIC, TAIX-Ray (authoritative E1 adapter), synthetic; pending adapters |
| `runner.py` | local runner: resumable shards over visible GPUs (no scheduler assumed) |
| `generators/` | one adapter per evaluated system (`base.py` contract; `hf.py` diffusers/transformers systems; `third_party.py` ProgEmu, UniMedVL, PRISM through their official code) |
| `images.py` | the one rendering rule from stored films to the 8-bit image every system receives |
| `resources/` | packaged YAML: constants, MIMIC item ontology, dependency table, system registry, prompt templates |

## Gate 0 tests

```bash
spectra selftest                      # or: python -m pytest -q tests
python scripts/fetch_tokenizers.py   # tokenizer files only, never weights (needs HF_HOME, HF_TOKEN)
HF_HUB_OFFLINE=1 SPECTRA_HF_TOKENIZERS=registry python -m pytest -q tests -k registered
```

- R001 `tests/test_r001_cer_demo.py` — CER compiler, cutoff, bundles and partition on the open MIMIC-IV demo 2.2
  (hand-checked windows of stays 33281088, 30757476, 32391858).
- R002 `tests/test_r002_exact_controls.py` — copy-prior C = 1/2; query-blind with shared reset seeds C = 1/2;
  bundle-blind JDA = 0; record-volume JDA = 0 on removal deletion and mirror; undefined channels.
- R003 `tests/test_r003_roundtrip.py` — round-trip and background equality for typed, event-text, verbal and
  compact interfaces; truncation and wrong bundles are caught; every registered text-interface system with its own tokenizer via `SPECTRA_HF_TOKENIZERS=registry`.
- `tests/test_components.py` — resolver R1–R5, device-state tiers, reference acceptance (point estimates + SD), PPI++, Holm table, export
  checks, splits, and reproduction of the measured TAIX-Ray funnel (3,567 eligible test patients).

Data used by the tests: the open MIMIC-IV demo 2.2 (`SPECTRA_MIMIC_DEMO`) and TAIX-Ray public metadata
(`SPECTRA_TAIX_META`); tests that need them are skipped when they are not staged.

## Scripts

| script | purpose |
|---|---|
| `scripts/fetch_tokenizers.py` | stage every registered system's tokenizer files (R003 offline) |
| `scripts/fetch_weights.py --machine cluster\|mimic` | stage the released weights of the systems that machine runs |
| `scripts/r004_smoke.py` | R004: one generation per system, each in its own env, through the certified interface |
| `scripts/r010_mimic_census.py` | R010: MIMIC film table, ICU stays, splits and the E1 funnel up to the extraction-free rules (counts only) |

Everything that runs on the MIMIC server, in order, is in [`MIMIC_SERVER.md`](MIMIC_SERVER.md).

## Status

The dependency table (`src/spectra/resources/dependency_table.yaml`) is a draft until the project clinician
signs it off at Gate 1. Adapters for ReXGradient, PadChest, MIDRC, BRAX, CheXpert Plus and NIH land when access is granted.
