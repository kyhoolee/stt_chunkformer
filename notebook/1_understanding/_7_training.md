Dưới đây là toàn bộ checklist đã “dỡ” hoàn toàn bảng ra các mục rõ ràng, chỉ dùng Markdown và code block để tránh lỗi hiển thị.

---

## 1. Dữ liệu (Data)

### 1.1 Raw-data audio

* **960 h Librispeech**

  * Download từ [http://www.openslr.org/12](http://www.openslr.org/12)
* **25 000 h Vietnamese (internal)**

  * Domains: Reading, Conversation, Telephony, YouTube

### 1.2 Dictionary

1. Gộp tất cả transcript (Anh + Việt) vào file `all_text.txt`.
2. Train BPE (vocab size = 5000) với SentencePiece:

   ```bash
   spm_train \
     --input=all_text.txt \
     --model_prefix=bpe5k \
     --vocab_size=5000 \
     --model_type=bpe
   ```
3. Kết quả: hai file `bpe5k.model` và `bpe5k.vocab`.

### 1.3 Augmented audio

* **Speed-Perturb**: speed factors = `[0.9, 1.0, 1.1]`
* **SpecAugment** trên filter-bank features:

  ```yaml
  # Frequency masking
  freq_mask:
    max_F: 27
    num_masks: 2

  # Time masking
  time_mask:
    max_T: 100
    num_masks: 2
  ```

---

## 2. Code & Hyper-parameters

### 2.1 Forward pass

* Luồng:

  1. `encoder_outputs = model.encoder(inputs)`
  2. `ctc_logits    = model.ctc_head(encoder_outputs)`
  3. `aed_logits    = model.aed_decoder(encoder_outputs, targets)`
* Chia module:

  * `chunkformer/model.py` chứa Encoder + Chunk-Attention + Conv-split
  * `chunkformer/heads.py` chứa `CTCHead` và `AEDDecoder`

### 2.2 Loss

```python
ctc_loss = F.ctc_loss(
    log_probs, targets, input_lengths, target_lengths
)
aed_loss = label_smoothed_cross_entropy(
    aed_logits, targets
)
total_loss = (
    lambda_ctc * ctc_loss
    + (1 - lambda_ctc) * aed_loss
)  # với lambda_ctc = 0.3
```

### 2.3 Optimizer & Scheduler

```python
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=peak_lr,
    betas=(0.9, 0.98),
    eps=1e-9
)
scheduler = NoamScheduler(
    optimizer,
    warmup_steps=warmup_steps
)
```

**Hyper-parameters**

* **Small-scale**

  * Epochs: 200 (≈ 400 000 steps)
  * Peak LR: 1e-3
  * Warm-up steps: 15 000
* **Large-scale**

  * Steps: 400 000
  * Peak LR: 1e-3
  * Warm-up steps: 25 000

---

## 3. Quy trình huấn luyện

1. **Full-context pre-training**

   ```bash
   python train.py \
     --data-config data/full_context.yaml \
     --model-config config/encoder_large.yaml \
     --epochs 200 \
     --lr 1e-3 \
     --warmup 15000 \
     --save-dir ckpt/full_context
   ```

2. **Fine-tune limited-context (small-scale)**

   ```bash
   python train.py \
     --data-config data/small_scale.yaml \
     --model-config config/chunkformer_[128,64,128].yaml \
     --resume ckpt/full_context/best.pt \
     --epochs 50 \
     --lr 1e-5 \
     --warmup 5000 \
     --save-dir ckpt/limited_small
   ```

3. **Fine-tune limited-context (large-scale)**

   ```bash
   python train.py \
     --data-config data/large_scale.yaml \
     --model-config config/chunkformer_[128,64,128].yaml \
     --resume ckpt/full_context/best.pt \
     --steps 100000 \
     --lr 1e-5 \
     --warmup 10000 \
     --save-dir ckpt/limited_large
   ```

---

## 4. Tips & Tricks

* **Checkpoint Averaging**

  ```bash
  average_checkpoints \
    --inputs ckpt/full_context/*.pt \
    --num-ckpts 50 \
    --output ckpt/full_context/avg50.pt
  ```

* **Dynamic Context Training**

  * Trong mỗi batch, random chọn một cấu hình `[latt, c, r]` từ danh sách ví dụ `[(128,64,128), (256,128,128), …]`.
  * Implement trong `DataLoader` hoặc callback.

* **Reset Peak LR trước fine-tune**

  ```python
  for g in optimizer.param_groups:
      g['lr'] = 1e-5
  ```

* **Warm-up Scheduler (Noam)**

  * LR tăng tuyến tính đến peak trong `warmup_steps`, rồi decay theo `step^{-0.5}`.

---

Dưới đây là ước tính thời gian cho từng giai đoạn huấn luyện trên cụm **8×NVIDIA H100**, sử dụng WeNet 2.0 với mixed-precision. Tất cả con số là tham khảo và có thể thay đổi tùy batch-size, độ dài trung bình của audio, và các chi tiết I/O.

| Giai đoạn                                             | Steps / Epochs             | Thời gian ước tính             | Ghi chú                                             |
| ----------------------------------------------------- | -------------------------- | ------------------------------ | --------------------------------------------------- |
| **1. Full-context pre-train (small-scale, 960 h LS)** | 200 epochs (\~400 k steps) | **50 – 65 giờ** (\~2–2.5 ngày) | Dữ liệu Librispeech; sequence ngắn → throughput cao |
| **2. Full-context pre-train (large-scale, 25 000 h)** | \~400 k steps              | **60 – 75 giờ** (\~2.5–3 ngày) | Audio dài hơn, I/O nặng hơn nên chậm hơn \~15%      |
| **3. Fine-tune limited-context (small-scale)**        | 50 epochs (\~100 k steps)  | **12 – 15 giờ**                | Resume từ full-context small; lr=1e-5               |
| **4. Fine-tune limited-context (large-scale)**        | 100 k steps                | **15 – 18 giờ**                | Resume từ full-context large; lr=1e-5               |

**Tổng cộng**: \~137–173 giờ (\~5.5–7 ngày) để hoàn thiện toàn bộ pipeline từ pre-train đến fine-tune.

> **Lưu ý**
>
> * Nếu dùng batch-size lớn hơn hoặc kinh nghiệm I/O tốt (caching, prefetch), bạn có thể rút ngắn \~10–20%.
> * Theo dõi utilization GPU; nếu thấp có thể tăng batch-size hoặc gradient-accumulation để đạt throughput tối ưu.
> * Phần checkpoint averaging và các bước hậu xử lý (evaluation, export) cũng chiếm thêm \~2–4 giờ.
