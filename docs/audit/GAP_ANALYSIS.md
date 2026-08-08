# XRFM — Gap Analysis (Phase 10)

Format: **Area | Current XRFM | Expected standard | Severity | Evidence | Fix**
Severity: CRITICAL / HIGH / MEDIUM / LOW. Evidence cites findings from `docs/audit/FORENSIC_AUDIT.md` (F-IDs) — all OBSERVED unless noted.

## Correctness Bugs

| Area | Current XRFM | Expected standard | Sev | Evidence | Fix |
|---|---|---|---|---|---|
| Causal masking | Implicit via SDPA import; manual path unmasked; zero tests | Explicit causal mask built in, tested, correct in every path | CRITICAL | F-01, F-02 | Build causal mask in MHA by default when `mask=None`; pass through to SDPA; add causality test |
| Tokenizer whitespace | `decode(encode(x)) != x`; no space tokens; newlines impossible | Round-trip-safe byte-level BPE preserving whitespace | CRITICAL | F-11 | Byte-level BPE (UTF-8), space-prefixed tokens, tests |
| Tokenizer Unicode | ValueError on non-Latin-1 | Full Unicode (byte-level) | CRITICAL | F-12 | Same as above |
| Mixed precision | GradScaler only; no autocast; claims bf16 | Either real AMP (autocast) or honest fp32 | CRITICAL | F-23 | Correct implementation or remove claim; keep NoOp on CPU |
| API import | `search_routes` missing → ImportError | `api.main` imports; server starts | CRITICAL | F-41 | Remove bad import (route exists in search module) |
| Loss on padding | Padded targets (id 0) contribute to loss | `ignore_index` masking | HIGH | F-15, F-39 | Ignore pad index in CE; mask in eval too |
| Vocab coherence | config 50304 vs tokenizer 408 vs script 1024 | Single source of truth | HIGH | F-13, F-18-BASELINE | Align config vocab with tokenizer; assert at startup |
| LR schedule resume | Scheduler state not checkpointed | Full state restoration | HIGH | F-24 | Add state_dict/load_state_dict to SchedulerLoader |
| Checkpoint resume | Committed ckpt has empty optimizer state | Resumable checkpoints | HIGH | F-31 | Save/restore optimizer + scheduler + config + seed |
| Init for depth | Xavier everywhere, no residual scaling | Scaled init for deep nets | MEDIUM | F-05 | Add optional scaled residual init (GPT-2 style) |
| BPE train/encode consistency | Merges applied in train can be missing in encode | Consistent merge record | MEDIUM | F-14 | Record every applied merge |

## Performance Problems

| Area | Current XRFM | Expected | Sev | Evidence | Fix |
|---|---|---|---|---|---|
| RoPE freq recompute | Recomputes cos/sin every call, ignores buffer | Precompute/cache | MEDIUM | F-03 | Use registered buffer; cache per (device,dtype) |
| Tokenizer efficiency | ~3.3–3.8 tokens/word general text | ~1.3–1.5 tokens/word | MEDIUM | F-16 | Byte-level BPE with merges across space-prefixed words |
| `generate_batch` | Recomputes prefix each step | KV-cached batch decode | LOW | F-38 | Reuse cache or document |
| Benchmark no-op | CUDA timings no-op on CPU | Honest fallbacks | LOW | F-46 | Perf-timer utility |

## Scalability Limitations

| Area | Current XRFM | Expected | Sev | Evidence | Fix |
|---|---|---|---|---|---|
| Data pipeline | In-memory char-split single file; no sharding/streaming | Sharded, streamed, boundary-aware | HIGH | F-20, F-22 | Design interface; implement shard reader for real corpora |
| Dataset | One repeated sentence (toy) | Diverse real corpus | CRITICAL | F-17 | Ship real small corpus (e.g., public-domain slice) |
| Distributed | Untested scaffolding | Validated DDP at least | MEDIUM | F-34 | CPU gloo multi-proc test; document GPU status |
| Embedding size | 67% of params in 50 K-vocab embedding | Proportionate vocab | MEDIUM | F-13 | Real 2–8 K vocab for small models; tie weights |
| Memory | No gradient checkpointing / activation management | Optional at scale | LOW | — | Document; implement only when needed |

## Research Gaps

| Area | Current XRFM | Expected | Sev | Evidence | Fix |
|---|---|---|---|---|---|
| Evaluation | Intrinsic PPL/acc only; no benchmarks; pads counted | Val PPL + at least one external sanity (e.g., WikiText slice) | HIGH | F-39, F-40 | Fix PPL masking; add WikiText-slice eval |
| Validation during training | None | Periodic val loss | HIGH | F-27 | Add eval step to loop |
| Scaling evidence | None | ≥2 size points with matched data | MEDIUM | — | Phase 20 mini scaling run |
| Baseline comparison | Claims "10M preset" vs 19.2 M actual | Documented true counts | MEDIUM | benchmark/model_forward.py | Fix naming; record real counts |

## Documentation Problems

| Area | Current XRFM | Expected | Sev | Evidence | Fix |
|---|---|---|---|---|---|
| Version strings | 5+ different versions | One version | MEDIUM | BASELINE §3 | Single version field |
| False claims | "Production Ready", "BF16", "pre-trained", "1,000×", "A++++++" | Claims match evidence | HIGH | F-48 | Rewrite README/status; correct mixed-precision docs |
| Missing referenced file | `scripts/validate_training.py` referenced, never existed | Docs reference real files | MEDIUM | F-48 | Provide the script or remove reference |
| Checkpoint provenance | No log/config for committed ckpt | Full provenance or removal from claims | HIGH | F-31, F-33 | Document honestly; new checkpoints carry metadata |

## Missing Tests

| Area | Current XRFM | Expected | Sev | Evidence | Fix |
|---|---|---|---|---|---|
| Causal mask test | None | Assert -inf above diagonal | CRITICAL | F-01 | Add |
| Reference math tests | None | RoPE/RMSNorm/SwiGLU vs reference | HIGH | §4 FORENSIC | Add |
| Training decrease/overfit | None | Loss ↓ on fixed data | HIGH | §4 | Add |
| Resume test | None (scheduler restart) | Step/LR restored | HIGH | F-24 | Add |
| Tokenizer round-trip/unicode | None | encode→decode fidelity | HIGH | F-11/F-12 | Add |
| API import/start | None | TestClient health | HIGH | F-41 | Add |
| Padding loss masking | None | ignore_index verified | MEDIUM | F-15 | Add |
| Seeding determinism | None | Same seed → same loss | MEDIUM | F-25 | Add |

## Infrastructure Problems

| Area | Current XRFM | Expected | Sev | Evidence | Fix |
|---|---|---|---|---|---|
| CI mypy | `mypy ... \|\| true` | Fail on errors | MEDIUM | F-43 | Remove `\|\| true` |
| CI scope | 192 self-referential tests only | Include train/eval smoke | MEDIUM | F-43 | Add tiny CPU train job |
| Docker GPU | torch downgraded to CPU wheel | Correct cu wheel pin | MEDIUM | F-44 | Pin index-url after requirements |
| API model load | Random init; no checkpoint load | Loads real weights | HIGH | F-42 | Load latest checkpoint; honest status |
| API stop/stream | `stop` ignored | Implement or remove | LOW | F-37 | Implement basic stop check |

## Training Problems

| Area | Current XRFM | Expected | Sev | Evidence | Fix |
|---|---|---|---|---|---|
| Seeding | None | Full seed control | HIGH | F-25 | Seed util in loop + loader |
| LR 1e-3 / wd 0.01 | Off modern recipe | Scale-appropriate (3–6e-4, wd 0.1) | MEDIUM | MODEL_COMPARISON §2 | Config defaults per size |
| Metrics logging | `logging.info` only | CSV/jsonl history | MEDIUM | F-27 | Add metrics writer |
| Gradient NaN policy | Hard crash | Skip/log | LOW | F-26 | error_if_nonfinite configurable |
| `train_custom_model` vocab mismatch | 1024 tokenizer vs 50304 model | Coherent | HIGH | F-13 | Build model from tokenizer vocab |
