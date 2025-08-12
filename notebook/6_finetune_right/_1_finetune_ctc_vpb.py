
from chunkformer_vpb.training import train as fine_tune_main

CFG_PATH = "../../config/right_vpb_ctc_finetune_config.yaml"

# 🔥 Gọi đúng hàm run_train (đã chỉnh sửa ở bước 1)
fine_tune_main.run_train(
        cfg_path=CFG_PATH, smoke=True, smoke_ratio=1, eval_train=True, eval_ratio=1, device="cuda",
        resume_ckpt_path='./checkpoints_vpb_ctc/epoch10.pt'
    )

# nohup env PYTHONUNBUFFERED=1 python _1_finetune_ctc_vpb.py > train_0_ctc_vpb.log 2>&1 &
