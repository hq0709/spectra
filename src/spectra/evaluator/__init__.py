"""Automated image evaluator (FINAL_PROPOSAL.md, Training Plan): frozen RAD-DINO tokens and one head per task,
trained on non-MIMIC data only; the same weights score every source.

Lifecycle, in order: (E-a) train the heads (findings on TAIX-Ray train; devices on PadChest dev and RANZCR CLiP);
(E-b) validate on held-out real films of every source against the section-12 criteria and set presence thresholds
on each source's development films; (E-c) one prespecified adaptation of a failing head on that source; (E-d)
freeze weights and thresholds before Gate 2 and before any test generation; (E-e) calibrate on generated films.

Readouts: phi_c(x) = expected ordinal grade for findings; psi_d(x) = P(present) for devices.
Modules: encoder (RAD-DINO, needs torch + transformers), heads (torch), metrics (numpy only).
"""
