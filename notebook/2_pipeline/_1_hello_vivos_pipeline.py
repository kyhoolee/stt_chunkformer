# %%
# Cell 2: import các thứ cần thiết
import os
import sys
# nếu data.py nằm cùng thư mục notebook thì đảm bảo path
sys.path.append(os.getcwd())

from chunkformer_vpb.data.data import build_data

# Cell 3: chạy pipeline với config
config_path = "_1_1_config.yaml"
build_data(config_path)

# %%


# %%



