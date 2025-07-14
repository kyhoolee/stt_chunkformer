### 1  Nhắc lại vai trò của hai nhánh trong mô hình hybrid CTC-Attention

| Nhánh                       | Mục tiêu                                                    | Đầu vào **transcript** đúng chuẩn                                                                     | Vì sao?                                                                                                                                                                                                                     |
| --------------------------- | ----------------------------------------------------------- | ----------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **CTC**                     | Học căn chỉnh thô frame ↔ token. Không sinh “kết thúc câu”. | **Không có** `<sos>`/`<eos>`, chỉ gồm các token lời nói thực + (ngầm) **blank** (ID = 0).             | Hệ dynamic-programming của CTC tự chèn blank & cho phép lặp-lại; việc ép thêm `<sos>/<eos>` làm tăng chiều dài target & buộc CTC phải “thấy” hai ký tự *không xuất hiện trong tín hiệu âm*  →  khó align, tăng loss vô ích. |
| **AED / Attention decoder** | Học mô hình ngữ cảnh & sinh token tự hồi quy.               | **Phải có** `<sos>` ở đầu **(input)** và `<eos>` ở cuối **(target)** để biết khi bắt đầu & dừng sinh. | Chuẩn teacher-forcing: $\text{input}_t=[\text{sos},y_1,\dots,y_{t-1}] \rightarrow \text{target}_t = y_t$ và token cuối để mô hình học dừng.                                                                                 |

---

### 2  Code hiện tại đang làm gì?

1. **`GreedyTokenizer.text2labels`**

   ```python
   seq = [sos] + ids + [eos]
   ```

   ⇒  luôn *thêm* `<sos>` & `<eos>` vào **ys\_pad** (duy nhất một tensor).

2. **`ASRModel.ctc.forward`**

   * Nhận **ys\_pad** nguyên vẹn (đã có `<sos>/<eos>`).
   * CTCLoss sẽ cố align cả hai token đó với tín hiệu âm thanh.

3. **`ASRModel._calc_att_loss`**

   * Nhận cùng **ys\_pad**, **tự tách**

     ```python
     ys_in  = ys_pad[:, :-1]   # bỏ <eos>
     ys_out = ys_pad[:, 1:]    # bỏ <sos>
     ```
   * Đây đúng chuẩn teacher-forcing cho AED.

**Kết quả**

* Phần **att-loss** đúng.
* **CTC-loss** đang “chịu hậu quả” của hai token không hiện hữu trong audio → loss CTC cao hơn cần thiết, có thể làm tổng loss tệ & làm chậm converge.

---

### 3  Cách làm “chuẩn” hơn

```text
           DataLoader             compute_loss
wav  ──▶  features           ─┐   (per batch)
txt  ──▶  token_seq (y)      ├─▶  CTC :   y           (không sos/eos)
                             └─▶  AED :  concat(sos, y, eos)
```

1. **DataLoader**

   * Trả về `toks` **không có** sos/eos, kèm `tok_lens`.

2. **compute\_loss\_batch**

   ```python
   sos_tok = torch.full((B,1), model.sos, device=device)
   eos_tok = torch.full((B,1), model.eos, device=device)

   ys_in  = torch.cat([sos_tok, toks], dim=1)        # <sos> y1 … yN
   ys_out = torch.cat([toks,    eos_tok], dim=1)     # y1 … yN <eos>
   ys_lens = tok_lens + 1        # cả in/out cùng +1
   # CTC
   loss_ctc, _ = model.ctc(enc_out, enc_lens, toks, tok_lens)
   # AED
   loss_att, _ = model._calc_att_loss(enc_out, enc_mask, ys_in_out_pad, ys_lens)
   ```

> *Lưu ý*: `_calc_att_loss` của bạn hiện mong **ys\_pad chứa luôn sos/eos**. Bạn có thể:
>
> * Hoặc sửa `_calc_att_loss` để nhận `ys_in`, `ys_out` riêng (ESPnet style).
> * Hoặc giữ nguyên, bằng cách ghép `[sos]+y+[eos]` → `ys_pad` trước khi gọi.

---

### 4  TL;DR – Có “đúng” không?

| Thành phần | Trạng thái hiện tại                        | Nên sửa?                                     |
| ---------- | ------------------------------------------ | -------------------------------------------- |
| **CTC**    | Bị ép học `<sos>/<eos>` → **không tối ưu** | **Có.** Bỏ `<sos>/<eos>` ra khỏi target CTC. |
| **AED**    | Đúng chuẩn teacher-forcing.                | Giữ nguyên.                                  |

Sau khi bạn quyết định tuân thủ quy tắc này, ta mới refactor DataLoader & hàm loss cho gọn (chỉ thêm `<sos>/<eos>` ở đúng chỗ của AED).
