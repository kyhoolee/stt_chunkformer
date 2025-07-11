from chunkformer_vpb.data.data import run_pipeline

# Chạy test 20 sample, có progress bar
metrics = run_pipeline(
    "_2_config_test.yaml", # "_2_config_train.yaml",
    test=False, # True,
    # test_size=20,
    use_progress=True,
    save_meta_in_test=True
)


print(metrics)

# nohup python _2_vivos_pipeline_progress.py --config _2_config_test.yaml > _2_log_test_full.txt 2>&1 &