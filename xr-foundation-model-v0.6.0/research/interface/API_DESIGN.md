# XRFM — API Design (Interfaces Only)

This document defines stable interfaces. No implementations yet (except where already built: `ConfigLoader`).

Every interface must remain stable across versions. Changing an interface requires a major version bump (`vX.Y.Z` → `v(X+1).0.0`).

---

## 1. Tokenizer Interface (`tokenizer/interface.py`)

```python
from abc import ABC, abstractmethod
from typing import List, Union


class TokenizerInterface(ABC):
    """Stable interface for all tokenizer algorithms.

    Conceptual reference: Standard tokenizer patterns (Sennrich et al., Kudo, OpenAI tiktoken)
    Implementation: Original — designed for future algorithm swaps (BPE, SentencePiece, Unigram, TikToken-style).
    """

    @abstractmethod
    def encode(self, text: str, **kwargs) -> List[int]:
        """Convert text string to integer token sequence."""
        ...

    @abstractmethod
    def decode(self, tokens: List[int], **kwargs) -> str:
        """Convert integer token sequence back to text string."""
        ...

    @abstractmethod
    def vocab_size(self) -> int:
        """Return size of tokenizer vocabulary."""
        ...

    @abstractmethod
    def save(self, path: str) -> None:
        """Persist tokenizer vocabulary and state to file system."""
        ...

    @abstractmethod
    def load(self, path: str) -> None:
        """Load tokenizer vocabulary and state from file system."""
        ...
```

**Usage in dataset loader:**
```python
# data/loader.py (planned, not implemented)
# loader will receive tokenizer instance (any subclass of TokenizerInterface)
# without knowing whether it's BPE, SentencePiece, or future algorithm.
```

**Stability guarantee:** This interface will not change between v0.2.0 and v2.0.0 unless a major version bump is required.

---

## 2. Configuration Interface (`xrfm/config/loader.py` — Already Implemented)

**Interface:** `ConfigLoader`

- `__init__(config_path: str)` — load from YAML.
- `get(key_path: str, default)` — dot-notation access.
- `model_config()` — returns `ModelConfig` dataclass.
- `training_config()` — returns `TrainingConfig` dataclass.
- `ConfigPresets` — factory for model presets (`xrfm_10m`, `xrfm_100m`, `xrfm_1b`).

**Stability guarantee:** Config interface is core. Changes only via minor version updates (new preset profiles) or major version updates (breaking config key changes).

---

## 3. Model Architecture Interface (`model/` — Planned)

```python
# Planned interface (Phase 4 design)
# The model class will expose:
# - forward(input_ids: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor
# - config() -> ModelConfig (access to hyperparameters)
# - save_checkpoint(path: str) -> None
# - load_checkpoint(path: str) -> None
```

**Design principle:** The model interface separates architecture definition (`GPTModel`) from training logic (`training/loop.py`). This allows the same model to be trained, fine-tuned, or served without rewrites.

---

## 4. Dataset Interface (`data/` — Planned)

```python
# Planned abstract interface
class DatasetInterface(ABC):
    @abstractmethod
    def load(self, tokenizer: TokenizerInterface) -> "Dataset":
        ...
    @abstractmethod
    def split_train_val(self, val_ratio: float = 0.1):
        ...
```

**Stability guarantee:** Dataset loader depends only on `TokenizerInterface`, not on specific tokenizer algorithms.

---

## 5. Training Interface (`training/` — Planned)

```python
# Planned interface structure
# Training loop will not be a single monolithic function.
# Instead, it will be composed of:
# - ConfigLoader (config source)
# - DatasetInterface (data source)
# - Model (architecture)
# - Optimizer (AdamW)
# - Scheduler (cosine decay)
# - CheckpointManager (save/load)
# These components interact through stable interfaces, allowing any component to be replaced.
```

**Example:** If FSDP is adopted (optional enhancement), only the `CheckpointManager` and `Optimizer` interfaces need updates. The dataset loader, model, and scheduler remain unchanged.

---

## 6. Inference Interface (`inference/` — Planned)

```python
# Planned interface
class InferenceEngine(ABC):
    @abstractmethod
    def generate(
        self,
        prompt: Union[str, List[str]],
        max_new_tokens: int = 50,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        **sampling_params
    ) -> Union[str, List[str]]:
        ...
    @abstractmethod
    def stream_generate(self, prompt: str, **params):
        """Generate tokens one at a time for streaming APIs."""
        ...
```

**Stability guarantee:** Sampling parameters (`temperature`, `top_p`, `top_k`) are standard across LLM serving. Changing the inference engine (custom → vLLM) should not require changing the `generate()` interface.

---

## 7. Logging Interface (`utils/logging.py` — Planned)

```python
# Planned modular interface
# The logging system supports multiple backends:
# - CSV (required, core)
# - TensorBoard (optional enhancement)
# - Weights & Biases (optional enhancement)
# The interface abstracts the backend:
# logger.log_metric(name: str, value: float, step: int) -> None
# logger.save_config(config: Dict) -> None
```

**Stability guarantee:** Adding a new backend (e.g., MLflow) requires only a new implementation of the logging interface, not changes to training or evaluation code.

---

## 8. Checkpoint Interface (`training/checkpoint.py` — Planned)

```python
# Planned interface
class CheckpointManager:
    def save(
        self,
        path: str,
        model_state: Dict,
        optimizer_state: Dict,
        scheduler_state: Dict,
        scaler_state: Optional[Dict],
        step: int,
        token_count: int,
        dataset_version: str,
        git_commit: str,
        config_snapshot: Dict
    ) -> None:
        ...
    def load(self, path: str) -> CheckpointData:
        ...
```

**Stability guarantee:** Checkpoint format is designed for future scaling. Adding new fields (e.g., FSDP sharded states) extends the interface without breaking backward compatibility.

---

## Summary: Interface Stability Commitment

Every interface listed above is designed to be stable across minor versions (`v0.X.Z` → `v0.(X+1).0`). Major version changes (`vX.Y.Z` → `v(X+1).0.0`) are required only for breaking architectural changes (e.g., switching from decoder-only to encoder-decoder as default architecture).

The goal is that a user who writes a custom dataset loader, custom tokenizer, or custom evaluation script against `v0.2.0` interfaces will find that same script works unchanged against `v0.9.0` (with optional enhancements available but not required).
