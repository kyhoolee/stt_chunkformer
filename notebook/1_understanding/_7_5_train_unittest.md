Đúng rồi, bạn có thể bắt đầu với **AILAB-VNUHCM/vivos** để nhanh chóng thử nghiệm:

---

## 1. Lý do chọn vivos trước

* Chỉ \~15 h – nhỏ gọn, tải nhanh, xử lý nhanh.
* Phòng thu cá nhân nhưng có multi-speaker, tiếng Việt rõ – phù hợp để “warm-up” pipeline trước khi bổ sung noise.

---

## 2. Ví dụ pipeline tích hợp vivos

### a) Tải và chuẩn hoá

```bash
pip install datasets soundfile sox

python - <<EOF
from datasets import load_dataset
import soundfile as sf
import os

ds = load_dataset("AILAB-VNUHCM/vivos", split="train")
os.makedirs("data/vivos_8k", exist_ok=True)

for ex in ds:
    wav, sr = sf.read(ex["path"])
    # downsample về 8 kHz nếu cần
    if sr != 8000:
        import subprocess
        out_path = f"data/vivos_8k/{os.path.basename(ex['path'])}"
        subprocess.run(["sox", ex["path"], "-r", "8000", out_path])
    else:
        sf.write(f"data/vivos_8k/{os.path.basename(ex['path'])}", wav, sr)
EOF
```

### b) Cấu hình data/telephony\_finetune.yaml

```yaml
train:
  - path: data/vivos_8k/*.wav
    aug:
      - specaugment
      - speed_perturb (0.9, 1.1)
      - add_noise (SNR 5–15 dB)
      - add_reverb (RT60 50–150 ms)
val:
  - path: data/vivos_8k_val/*.wav
    aug: none
```

*(Bạn có thể chia 15 h thành train/val 90/10%)*

### c) Chạy fine-tune nhanh

```bash
python train.py \
  --data-config data/telephony_finetune.yaml \
  --model-config config/chunkformer_callbot.yaml \
  --resume path/to/paper_final_checkpoint.pt \
  --epochs 5 \
  --lr 5e-5 \
  --warmup 500 \
  --batch-size 16 \
  --save-dir ckpt/callbot_vivos
```

* Chạy 5 epochs để kiểm tra pipeline, nhìn curve loss/WER.
* Nếu OK, tăng lên 10–20 epochs hoặc thêm dữ liệu khác.

---

Bằng cách này bạn nhanh chóng validate end-to-end từ data → model → WER, trước khi mở rộng sang các tập lớn hơn.
