
from chunkformer_vpb.training import train as fine_tune_main

CFG_PATH = "../../config/aug_vpb_ctc_finetune_config.yaml"

# 🔥 Gọi đúng hàm run_train (đã chỉnh sửa ở bước 1)
# NOTE: aug_train_data lớn -> ko eval train
fine_tune_main.run_train(
        cfg_path=CFG_PATH, smoke=True, smoke_ratio=1, eval_train=False, eval_ratio=1, device="cuda",
        resume_ckpt_path=None
    )

# nohup env PYTHONUNBUFFERED=1 python _10_finetune_ctc_aug.py > train_0_ctc_aug.log 2>&1 &
