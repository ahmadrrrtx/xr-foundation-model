#!/usr/bin/env bash
# XRFM distributed training launch script (v0.8.0)
# Usage: torchrun --nproc_per_node=N scripts/torchrun_launch.sh
set -euo pipefail

echo "=== XRFM v0.8.0 — Distributed Training ==="
echo "RANK: ${RANK:-0}  LOCAL_RANK: ${LOCAL_RANK:-0}  WORLD_SIZE: ${WORLD_SIZE:-1}"
echo "MASTER_ADDR: ${MASTER_ADDR:-localhost}  MASTER_PORT: ${MASTER_PORT:-29500}"

python -c "
from training.distributed import init_distributed, get_rank, get_world_size, is_main_process
init_distributed()
if is_main_process():
    print(f'Distributed initialized: world_size={get_world_size()}')
    from model.gpt import GPTModel
    from training.loop import TrainingLoop
    from xrfm.data.loader import XRFMTextDataset
    from tokenizer.bpe import BytePairEncoder
    import torch
    print('Model + dataset OK — distributed environment ready')
    print(f'CUDA available: {torch.cuda.is_available()}')
    if torch.cuda.is_available():
        print(f'Device count: {torch.cuda.device_count()}')
"
