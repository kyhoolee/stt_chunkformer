Để nhanh chóng có được một bản **ChunkFormer** tối ưu cho kịch bản **call-bot (telephony speech)**, ta có thể **giản lược** quy trình fine-tune như sau:

---

## 1. Chọn dữ liệu ưu tiên

1. **Telephony (internal)**
   – 8 kHz, tiếng ồn đường dây, gần với môi trường call-bot thực tế.
2. **Conversation (internal)**
   – Hội thoại tự nhiên, tốc độ và ngắt quãng tương tự agent–customer.
3. **(Tuỳ chọn) Credit thêm một ít “Reading”**
   – Để ổn định mô hình với ngữ điệu rõ ràng, nhưng chỉ 10–20 % so với tổng data.

> **Không cần dùng đến** Librispeech hay YouTube cho bước fine-tune này – chúng quá “sạch” hoặc quá đa dạng domain.

---

## 2. Chọn augmentation ưu tiên

1. **SpecAugment**

   * Giữ nguyên config cũ: 2 masks cho freq (max F=27) + 2 masks cho time (max T=100).
2. **Speed-Perturb**

   * Giữ 0.9 & 1.1 để mô phỏng giọng khách hàng nhanh/chậm.
3. **Telephony noise augmentation**

   * Thêm tiếng ồn thực (white-noise hoặc real call-center noise) với SNR từ 5–15 dB.
   * Thêm echo/reverb nhẹ (RT60 ≈ 50–150 ms).

---

## 3. Công việc fine-tune rút gọn

1. **Chuẩn bị data config**

   * Combine Telephony 80 % + Conversation 20  % (+Reading 10 % nếu cần)
   * Augmentation inline trong DataLoader.

2. **Nạp checkpoint gốc**

   ```bash
   --resume path/to/paper_final_checkpoint.pt
   ```

3. **Chạy fine-tune**

   ```bash
   python train.py \
     --data-config data/telephony_finetune.yaml \
     --model-config config/chunkformer_[128,64,128].yaml \
     --resume path/to/paper_final_checkpoint.pt \
     --epochs 20 \
     --lr 5e-5 \
     --warmup 2000 \
     --batch-size 32 \
     --save-dir ckpt/callbot
   ```

   * **Epochs**: 20 (hoặc dừng sớm khi WER trên validation plateau)
   * **LR**: 5 × 10⁻⁵
   * **Warm-up**: 2 000 steps
   * **Batch-size**: chọn sao cho GPU \~95 % utilization (có thể 16–32 với H100).

4. **Checkpoint Averaging (tuỳ chọn)**

   ```bash
   average_checkpoints \
     --inputs ckpt/callbot/*.pt \
     --num-ckpts 5 \
     --output ckpt/callbot/avg5.pt
   ```

---

### Lý do rút gọn

* **Chỉ tập trung domain “call-bot”** (telephony + conversation) để mô hình học đúng noise, tốc độ, ngắt quãng.
* **Giảm số epoch** và LR cao hơn tí so với fine-tune thông thường để nhanh converge.
* **Augmentation noise + echo** giúp mô phỏng thực tế call center.

Với pipeline này bạn có thể ra nhanh một bản **ChunkFormer** sẵn sàng chạy trên audio call-bot trong **1–2 ngày** thay vì hàng tuần với full-scale fine-tune.


-----------------------

Đúng rồi – chúng ta sẽ **bỏ hoàn toàn** các bước pre-train và fine-tune quy mô lớn cũ, chỉ tập trung vào một **bước duy nhất**:

---

# Fine-tune nhanh cho Call-Bot

### 1. Chuẩn bị

* **Checkpoint**: `paper_final_checkpoint.pt`
* **Data**: chỉ lấy

  * Telephony (80 %)
  * Conversation (20 %)

### 2. Config đơn giản

```yaml
# data/telephony_finetune.yaml
train:
  - path: /data/telephony/*.wav
    aug: 
      - specaugment
      - speed_perturb (0.9, 1.1)
      - add_noise (SNR 5–15 dB)
      - add_reverb (RT60 50–150 ms)
val:
  - path: /data/telephony_val/*.wav
    aug: none
```

```yaml
# config/chunkformer_callbot.yaml
model:
  context: [128,64,128]
optim:
  lr: 5e-5
  warmup_steps: 2000
trainer:
  epochs: 10
  batch_size: 32
  resume_checkpoint: paper_final_checkpoint.pt
  save_dir: ckpt/callbot
```

### 3. Chạy fine-tune

```bash
python train.py \
  --data-config data/telephony_finetune.yaml \
  --model-config config/chunkformer_callbot.yaml
```

* **Epochs**: 10 (dừng sớm khi val-WER không giảm)
* **LR**: 5 × 10⁻⁵, warm-up 2 000 steps
* **Batch-size**: 32 (tùy GPU)

### 4. (Tuỳ chọn) Averaging

```bash
average_checkpoints \
  --inputs ckpt/callbot/*.pt \
  --num-ckpts 3 \
  --output ckpt/callbot/avg3.pt
```

---

**Kết quả**: chỉ trong **1 bước fine-tune** (1–2 ngày), bạn có model đã điều chỉnh đặc biệt cho audio call-bot mà không cần lặp lại toàn bộ pipeline phức tạp trước đó.


