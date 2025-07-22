
from chunkformer_vpb.training import train as fine_tune_main

CFG_PATH = "../../config/vpb_cmvn_finetune_config.yaml"

# 🔥 Gọi đúng hàm run_train (đã chỉnh sửa ở bước 1)
fine_tune_main.run_train(
        cfg_path=CFG_PATH, smoke=True, smoke_ratio=1, eval_train=True, eval_ratio=1, device="cuda",
        resume_ckpt_path=None
    )

# nohup env PYTHONUNBUFFERED=1 python _6_finetune_resume_vpb.py > train_0_resume_vpb.log 2>&1 &
