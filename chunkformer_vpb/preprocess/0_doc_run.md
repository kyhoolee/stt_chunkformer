tar -xf manifests.tar
tar -xf voice_data_8k.tar


python -m chunkformer_vpb.preprocess._5_1_small_ds_parallel --mode debug --dataset vietmed
