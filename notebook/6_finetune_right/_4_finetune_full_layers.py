
from chunkformer_vpb.training import train as fine_tune_main

CFG_PATH = "../../config/right_vpb_full_layers.yaml"

# 🔥 Gọi đúng hàm run_train (đã chỉnh sửa ở bước 1)
fine_tune_main.run_train(
        cfg_path=CFG_PATH, smoke=False, smoke_ratio=1, eval_train=True, eval_ratio=1, device="cuda",
        resume_ckpt_path=None  # @NOTE: Không cần resume_ckpt_path vì đây là huấn luyện from start
    )

# nohup env PYTHONUNBUFFERED=1 python _4_finetune_full_layers.py > train_4_full_layer.log 2>&1 &
