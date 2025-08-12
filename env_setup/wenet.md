Dưới đây là câu lệnh chuẩn để **install environment từ file `.yml`** bằng **micromamba**:

```bash
micromamba create -n stt310 -f stt310_simple.yml
```

### Ví dụ cụ thể:

```bash
micromamba create -n stt_env -f environment.yml
```

### Giải thích:

* `-n stt_env`: đặt tên cho environment là `stt_env`
* `-f environment.yml`: chỉ định file cấu hình environment

---

### Nếu muốn **tự động activate** sau khi cài:

```bash
micromamba create -n stt_env -f environment.yml && micromamba activate stt_env
```

### Nếu bạn đã có file `.yml` nằm ở đường dẫn cụ thể:

```bash
micromamba create -n stt_env -f /path/to/your/env.yml
```

---



- wenet
pip install git+https://github.com/wenet-e2e/wenet.git

- chunkformer_vpb
pip install -e .

- install other 
pip install torchaudio