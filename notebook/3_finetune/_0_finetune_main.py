from chunkformer_vpb.training import train as fine_tune_main

# fine_tune_main.train()


from chunkformer_vpb.training import train as fine_tune_main

CFG_PATH = "../../config/vpb_finetune_config.yaml"

# 🔥 Gọi đúng hàm run_train (đã chỉnh sửa ở bước 1)
fine_tune_main.run_train(cfg_path=CFG_PATH, smoke=True, eval_train=True)



# dùng mặc định 1%
# python fine_tune_main.py --config ../../config/finetune_config.yaml --smoke

# # hoặc test nhanh hơn nữa: 0.005 = 0.5% data
# python fine_tune_main.py --config ../../config/finetune_config.yaml --smoke --smoke-ratio 0.005
# nohup python _0_finetune_main.py --config ../../config/finetune_config.yaml --smoke --smoke-ratio 0.1 > train_smoke.log 2>&1 &
