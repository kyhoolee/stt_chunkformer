
from chunkformer_vpb.training import train as fine_tune_main

CFG_PATH = "../../config/vpb_chunk_finetune_config.yaml"

# 🔥 Gọi đúng hàm run_train (đã chỉnh sửa ở bước 1)
fine_tune_main.run_train(
        cfg_path=CFG_PATH, smoke=True, smoke_ratio=1, eval_train=True, eval_ratio=1, device="cuda",
        resume_ckpt_path=None
    )

# nohup env PYTHONUNBUFFERED=1 python _9_finetune_chunk_vpb.py > train_0_chunk_vpb.log 2>&1 &
