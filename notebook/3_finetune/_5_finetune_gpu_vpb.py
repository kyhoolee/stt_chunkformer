
from chunkformer_vpb.training import train as fine_tune_main

CFG_PATH = "../../config/vpb_gpu_finetune_config.yaml"

# 🔥 Gọi đúng hàm run_train (đã chỉnh sửa ở bước 1)
fine_tune_main.run_train(cfg_path=CFG_PATH, smoke=True, smoke_ratio=1, eval_train=True, eval_ratio=1, device="cuda")

# nohup env PYTHONUNBUFFERED=1 python _5_finetune_gpu_vpb.py > train_full_gpu_vpb.log 2>&1 &
