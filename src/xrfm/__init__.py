"""
XRFM — XR Foundation Model.

A small, coherent, pure-PyTorch foundation-model research framework:
decoder-only transformer (RoPE/RMSNorm/SwiGLU), byte-level BPE tokenizer,
line-boundary data pipeline, validated typed configuration, trainer,
KV-cached inference, and perplexity/accuracy evaluation.

Public API (the only import surface users should need)::

    import xrfm

    xrfm.__version__                      # single version source
    cfg = xrfm.load_config("config/tiny.yaml")   # or xrfm.load_config()
    model = xrfm.XRFM(cfg)                # the transformer (alias: XRFMModel)
    tok = xrfm.BPETokenizer.pretrained()  # packaged tokenizer resource
    ds = xrfm.TextDataset(cfg.data, tok, split="train")
    trainer = xrfm.Trainer(model, config=cfg)
    out = trainer.train(ds)
    result = xrfm.generate(model, "Once upon a time", tokenizer=tok)
    metrics = xrfm.evaluate(model, val_loader)

Everything else (``xrfm.models.layers.*``, ``xrfm.training.loop``, ...)
is internal and may change; see ``docs/architecture.md`` for the module
map and ``docs/adr/0001-xrfm-core-architecture.md`` for the reasoning.

Secondary/experimental subsystems (RAG search, optimization utilities,
the NeuroTopo research model) are importable but clearly marked and are
consumers of this core — never the other way around.
"""

__version__ = "1.0.1"
__author__ = "XR Foundation Model Contributors"

__all__ = [
    # meta
    "__version__",
    "__author__",
    # configuration
    "ConfigError",
    "ConfigLoader",
    "DatasetConfig",
    "ModelConfig",
    "TrainingConfig",
    "XRFMConfig",
    "default_config",
    "load_config",
    # model
    "GPTModel",
    "XRFM",
    "XRFMEmbedding",
    "XRFMModel",
    # tokenization
    "BPETokenizer",
    "BytePairEncoder",
    "Tokenizer",
    "TokenizerInterface",
    "decode_text",
    "encode_text",
    # data
    "Dataset",
    "TextDataset",
    "XRFMTextDataset",
    # training
    "TrainResult",
    "Trainer",
    "TrainingLoop",
    # inference
    "GenerationEngine",
    "GenerationResult",
    "generate",
    # evaluation
    "compute_perplexity",
    "evaluate",
]

# ---------------------------------------------------------------------
# config
# ---------------------------------------------------------------------
from xrfm.config import (  # noqa: E402
    ConfigError,
    ConfigLoader,
    DatasetConfig,
    ModelConfig,
    TrainingConfig,
    XRFMConfig,
    default_config,
    load_config,
)

# ---------------------------------------------------------------------
# data
# ---------------------------------------------------------------------
from xrfm.data import TextDataset, XRFMTextDataset  # noqa: E402

# ---------------------------------------------------------------------
# evaluation
# ---------------------------------------------------------------------
from xrfm.evaluation import compute_perplexity, evaluate  # noqa: E402

# ---------------------------------------------------------------------
# inference
# ---------------------------------------------------------------------
from xrfm.inference import GenerationEngine, GenerationResult, generate  # noqa: E402

# ---------------------------------------------------------------------
# model
# ---------------------------------------------------------------------
from xrfm.models import GPTModel, XRFMEmbedding, XRFMModel  # noqa: E402

# ---------------------------------------------------------------------
# tokenization
# ---------------------------------------------------------------------
from xrfm.tokenization import (  # noqa: E402
    BPETokenizer,
    BytePairEncoder,
    Tokenizer,
    TokenizerInterface,
    decode_text,
    encode_text,
)

# ---------------------------------------------------------------------
# training
# ---------------------------------------------------------------------
from xrfm.training import Trainer, TrainingLoop, TrainResult  # noqa: E402

# Canonical short name for the model class:
#     model = XRFM(config)
# ``XRFMModel`` is the same class; ``GPTModel`` is the historical name.
XRFM = XRFMModel

# Historical/convenience alias for the dataset class.
Dataset = TextDataset
