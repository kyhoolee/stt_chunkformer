Dưới đây là **quy trình “smoke-test training”** kèm đoạn mã gợi ý để bạn cắm thẳng vào `train.py` (hoặc chạy riêng `train_debug.py`). Mục tiêu: chạy vài bước đầu, in chi tiết mọi giá trị quan trọng để nhìn ra lỗi shape/NaN/gradient ngay lập tức.

---

## 1. Bật **chế độ debug** nhẹ nhàng

```python
# thêm ở đầu train.py
import torch, math, os
torch.autograd.set_detect_anomaly(True)           # dò NaN/Inf trong backward

DEBUG_STEPS = int(os.getenv("DEBUG_STEPS", 20))   # chỉ chạy n bước đầu
PRINT_EVERY = 1                                   # in log mỗi step
```

---

## 2. In thông tin batch + loss

```python
for step, (feats, feat_lens, toks, tok_lens) in enumerate(train_loader, 1):
    ...
    loss, loss_ctc, loss_att = compute_loss_batch(...)

    # ---- LOG chi tiết ----
    if step <= DEBUG_STEPS and step % PRINT_EVERY == 0:
        b, t_max, _ = feats.shape
        l_max = toks.shape[1]
        print(f"[DBG] step={step:3d}  B={b}  T_max={t_max}  L_max={l_max} "
              f"loss={loss.item():.4f}  ctc={loss_ctc.item():.4f}  att={loss_att.item():.4f}")

        # grad-norm
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), math.inf)
        print(f"       grad_norm={grad_norm:.2f}  lr={sched.get_last_lr()[0]:.6e}")

        # kiểm NaN
        if torch.isnan(loss):
            raise ValueError("❌  NaN loss detected!")
```

> **TIP**: sau 20 bước debug, nếu mọi thứ ổn bạn có thể `export DEBUG_STEPS=0` để tắt.

---

## 3. Giới hạn dữ liệu (mini-subset)

Create file `subset.yaml` với:

```yaml
training:
  epochs: 1
  batch_size: 2
  shuffle: false
data:
  manifest_dir: ./cache_train_subset   # chỉ chứa 50 mẫu
```

Hoặc chạy nhanh qua CLI:

```bash
python train.py --config finetune_config.yaml \
                --epochs 1 \
                --subset 50   # nếu bạn tự thêm arg
```

---

## 4. Kiểm tra LR & clip-grad

Trong log debug ở bước 2 bạn đã in `grad_norm` và `lr`. Quan sát:

```
[DBG] step=  1  B=2  T_max=780  L_max=60 loss=5.1234  ctc=3.76  att=6.48
       grad_norm=2.91  lr=9.999999e-06
[DBG] step= 10 ... lr=2.000000e-04      # warm-up lên đỉnh
```

* **grad\_norm** quá cao → tăng `max_grad_norm` hoặc giảm LR.
* **lr** không thay đổi → scheduler chưa `step()`.

---

## 5. Kiểm tra shape encoder

Nếu nghi ngờ `_chunk_encoder_forward`:

```python
enc_out, enc_mask = _chunk_encoder_forward(feats[:1], model, cfg.chunk, device)
print("[DBG] encoder_out:", enc_out.shape, " mask_sum:", enc_mask.sum().item())
```

`mask_sum` phải bằng `enc_out.shape[1]`.

---

## 6. Chạy Eval sau 20 bước

```python
if step == DEBUG_STEPS:
    print("=== Smoke eval ===")
    evaluate(model, tokenizer, valid_loader, cfg, device)
    break   # thoát sớm
```

---

### Kết quả mong đợi trong smoke-test

```
[DBG] step= 20  B=2  T_max=820  L_max=62 loss=3.87  ctc=1.45  att=5.12
       grad_norm=2.30  lr=1.8e-04
=== Smoke eval ===
== Dev WER: 38.21 %
```

Chỉ cần loss hữu hạn và WER ra con số (dù cao) là pipeline thông luồng.

---

**Nếu gặp lỗi**

* **NaN/Inf** → check chuẩn hoá FBANK, text, hoặc giảm LR.
* **shape mis-match** → print `.shape` trước dòng báo lỗi để xác định tensor nào sai.
* **CUDA out of memory** → giảm `batch_size`/`chunk_size`.

Dán traceback cụ thể, mình sẽ hỗ trợ fix tiếp. Chúc smoke-test suôn sẻ!
