Dưới đây là **tổng hợp** các phần code bạn cần bổ sung vào script `train.py` (hoặc module fine-tune riêng) để “sẵn sàng” cho bước fine-tuning CTC + AED như đã thảo luận:

---

## 1. Định nghĩa Loss

```python
import torch.nn as nn

# Blank token ID từ dictionary
blank_id = char_dict["<blk>"]

# CTC loss
ctc_criterion = nn.CTCLoss(blank=blank_id, zero_infinity=True)

# AED loss (CrossEntropy, ignore padding)
aed_criterion = nn.CrossEntropyLoss(ignore_index=-100)

# Cân bằng hai loss
lambda_ctc = 0.3
```

---

## 2. Khởi tạo Optimizer & Scheduler

```python
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup

# Hyper-params
lr = 5e-5
warmup_steps = 2000
total_steps = num_epochs * steps_per_epoch

# Optimizer
optimizer = AdamW(
    model.parameters(),
    lr=lr,
    betas=(0.9, 0.98),
    eps=1e-9,
    weight_decay=0.01
)

# Scheduler (linear warmup + decay)
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=total_steps
)
```

---

## 3. Training Step Function

```python
import torch

def train_step(batch):
    feats, feat_lens, targets, tgt_lens = batch
    feats, feat_lens = feats.to(device), feat_lens.to(device)
    targets, tgt_lens = targets.to(device), tgt_lens.to(device)

    # Forward
    encoder_out = model.encoder(feats)            # [B, T', H]
    ctc_logits  = model.ctc(encoder_out)          # [B, T', V]
    aed_logits  = model.decoder(
        encoder_out,
        targets[:, :-1]
    )                                             # [B, L, V]

    # CTC loss
    log_probs = ctc_logits.log_softmax(-1).permute(1, 0, 2)
    loss_ctc = ctc_criterion(
        log_probs, targets, feat_lens, tgt_lens
    )

    # AED loss
    b, l, v = aed_logits.size()
    loss_aed = aed_criterion(
        aed_logits.view(-1, v),
        targets[:, 1:].contiguous().view(-1)
    )

    # Total loss
    loss = lambda_ctc * loss_ctc + (1 - lambda_ctc) * loss_aed

    # Backward + update
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
    optimizer.step()
    scheduler.step()

    return loss.item(), loss_ctc.item(), loss_aed.item()
```

---

## 4. DataLoader & Augmentation

* **DataLoader** phải trả về batch `(feats, feat_lens, targets, tgt_lens)`
* **Augment** tích hợp ngay trong `Dataset.__getitem__` hoặc collate:

  * SpecAugment (2 freq masks, max F=27; 2 time masks, max T=100)
  * Speed-Perturb (0.9, 1.1)
  * Add Noise (SNR 5–15 dB), Add Reverb (RT60 50–150 ms)

---

## 5. Chạy Fine-tune

```bash
python train.py \
  --data-config data/telephony_finetune.yaml \
  --model-config config/chunkformer_callbot.yaml \
  --resume path/to/paper_final_checkpoint.pt
```

* epochs: 10–20 (dừng sớm khi val-WER plateau)
* batch-size: 16–32 (tối ưu GPU utilization)

---

### Checklist hoàn chỉnh

1. Load model + chuyển `model.train()`
2. Định nghĩa `ctc_criterion`, `aed_criterion`, `lambda_ctc`
3. Khởi tạo `optimizer`, `scheduler`
4. Implement `train_step()` như bên trên
5. DataLoader trả về đúng 4 tensor và áp augmentation
6. Viết loop epoch/step, save checkpoint, validate WER
7. (Tuỳ chọn) Checkpoint averaging sau khi training xong

---------------------------------------

Dưới đây là bảng tổng hợp các siêu-tham số (“hyper-parameters”) chính cho bước fine-tune Call-Bot, cùng giá trị đề xuất, so sánh với paper gốc và giải thích lựa chọn:

| Thành phần                        | Giá trị đề xuất                           | Paper gốc                                | Giải thích                                                                                          |
| --------------------------------- | ----------------------------------------- | ---------------------------------------- | --------------------------------------------------------------------------------------------------- |
| **λ (CTC vs AED)**                | 0.3                                       | 0.3 (hybrid loss)                       | Giữ nguyên theo paper để cân bằng tốt giữa alignment-free CTC và ngữ cảnh của AED.                  |
| **LR (peak learning rate)**       | 5 × 10⁻⁵                                  | 1 × 10⁻³ (pre-train)                     | Fine-tune domain hẹp chỉ cần LR nhỏ, tránh “xoá nhoè” các trọng số đã học; 5e-5 là mức thường dùng. |
| **Warm-up steps**                 | 2 000                                     | 15 000 (small), 25 000 (large)           | Vì chỉ train \~10–20 epochs nên warm-up ngắn cho nhanh đạt peak LR, rồi decay để ổn định.           |
| **Scheduler**                     | Linear Warm-up + decay                    | Noam (warm-up + inverse-sqrt)            | Linear warm-up + decay đơn giản, dễ implement; với fine-tune nhỏ, Noam chưa cần thiết.              |
| **Optimizer**                     | AdamW                                     | Adam + Noam scheduler                    | AdamW thêm weight decay giúp regularize khi fine-tune, hạn chế over-fit với tập nhỏ hơn.            |
| **Weight decay**                  | 0.01                                      | 0 (không đề cập)                         | Thêm decay nhẹ để giảm over-fit domain hạn hẹp.                                                     |
| **Batch size**                    | 16–32                                     | không cố định                            | Giá trị tùy GPU (H100), đủ lớn để tận dụng throughput mà không tràn OOM.                            |
| **Epochs**                        | 10–20                                     | 50 epochs (small), 100 000 steps (large) | Fine-tune nhanh: 10–20 epochs (dừng sớm khi val-WER plateau) thay vì chạy long-scale.               |
| **Gradient clipping (max\_norm)** | 5.0                                       | không đề cập                             | Giúp ổn định training, tránh bùng nổ gradient khi fine-tune với LR tương đối cao so với scale nhỏ.  |
| **SpecAugment**                   | freq\_mask F≤27 (2), time\_mask T≤100 (2) | như paper                                | Giữ nguyên config proven của paper.                                                                 |
| **Speed-Perturb**                 | factors \[0.9, 1.1]                       | \[0.9, 1.0, 1.1] (pre-train)             | Loại bỏ “clean” 1.0 để tập trung augment biến thể; vẫn giữ hai tốc độ 0.9/1.1 để model thích ứng.   |
| **Noise/Reverb augmentation**     | SNR 5–15 dB, RT60 50–150 ms               | không đề cập                             | Thêm domain-specific augment cho call-bot (đường dây, echo), giúp đánh giá thực tế tốt hơn.         |

### Lý do chính cho từng lựa chọn

1. **Giữ λ = 0.3** để đảm bảo mô hình vẫn kết hợp được cả alignment-free CTC (tốt với audio cắt khúc) và AED (tốt với ngữ cảnh dài).
2. **LR nhỏ (5e-5)**: quá trình fine-tune không cần learning rate cao như pre-train; 5e-5 là mức phổ thông giúp nhanh hội tụ mà không phá vỡ trọng số gốc.
3. **Warm-up ngắn**: fine-tune chỉ vài chục nghìn bước, không cần warm-up quá lâu; 2 000 bước giúp LR lên peak đủ nhanh.
4. **AdamW + weight decay**: weight decay 0.01 giúp regularize khi fine-tune với data ít và domain hẹp, giảm over-fit.
5. **Batch size 16–32**: ưu tiên throughput cao trên H100, nhưng cần thử để không OOM do model \~110 M params + attention context.
6. **Gradient clipping**: dù paper không nhắc, nhưng với LR còn cao so với bước fine-tune ngắn, clipping max\_norm=5 giúp training ổn định.
7. **Augmentation domain-specific**: giữ SpecAugment của paper, giảm speed-perturb một level, và thêm noise/reverb để mô phỏng đúng môi trường call-bot.



