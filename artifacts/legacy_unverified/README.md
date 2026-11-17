# Quarantined Legacy Unverified Artifacts

Files in this directory are **historical pre-audit artifacts** moved out of active runtime discovery paths (`checkpoints/`) during **Phase 0 (`GATE-0`)** stabilization:

- `checkpoint_step_500.pt` (76.8 MB): Pre-audit checkpoint (`vocab_size=50304`, `d_model=256`, `n_layers=6`, empty `optimizer_state_dict`, no scheduler or manifest hash). Retained solely for historical audit reproducibility (`docs/audit/BASELINE.md`); **never loaded by XRFM runtime or API**.
