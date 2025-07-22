
from chunkformer_vpb.training import train as fine_tune_main

CFG_PATH = "../../config/vpb_finetune_config.yaml"

# 🔥 Gọi đúng hàm run_train (đã chỉnh sửa ở bước 1)
fine_tune_main.run_train(cfg_path=CFG_PATH, smoke=True, smoke_ratio=1, eval_train=True, eval_ratio=1)

# nohup env PYTHONUNBUFFERED=1 python _4_finetune_full_vpb.py > train_full_0.25_vpb.log 2>&1 &
# nohup env PYTHONUNBUFFERED=1 python _4_finetune_full_vpb.py > train_full_1_vpb.log 2>&1 &
