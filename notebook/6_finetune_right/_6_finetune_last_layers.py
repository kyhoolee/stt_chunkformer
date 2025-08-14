
from chunkformer_vpb.training import train as fine_tune_main

CFG_PATH = "../../config/right_clean_last_layer.yaml"

# 🔥 Gọi đúng hàm run_train (đã chỉnh sửa ở bước 1)
fine_tune_main.run_train(
        cfg_path=CFG_PATH, smoke=False, smoke_ratio=1, eval_train=True, eval_ratio=1, device="cuda",
        resume_ckpt_path=None  # @NOTE: Không cần resume_ckpt_path vì đây là huấn luyện from start
    )

# nohup env PYTHONUNBUFFERED=1 python _6_finetune_last_layers.py > train_6_last_layer.log 2>&1 &
