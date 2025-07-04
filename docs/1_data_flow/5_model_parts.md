🧮 === MODEL PARAMETER SUMMARY ===
🔎 Full Model
   • Total parameters       : 113,852,240
   • Trainable parameters   : 113,852,240
   • Frozen parameters      : 0
   • Estimated size (float32): 434.31 MB

🔎 Encoder
   • Total parameters       : 113,852,240
   • Trainable parameters   : 113,852,240
   • Frozen parameters      : 0
   • Estimated size (float32): 434.31 MB

🔎 CTC Head
   • Total parameters       : 3,586,896
   • Trainable parameters   : 3,586,896
   • Frozen parameters      : 0
   • Estimated size (float32): 13.68 MB

✅ Done! Saved model structure to folder: model_architect

----------------------------------------


Đúng rồi, bạn đã phát hiện chuẩn — việc **Full Model** và **Encoder** có số lượng tham số giống hệt nhau (`113,852,240`) chứng tỏ **CTC Head không nằm trong `model` chính**, hoặc đã được tách ra và không được gắn vào mô hình như một `nn.Module` con.

### 📌 Nguyên nhân có thể:

#### 1. `self.ctc` không được `register_module`

Trong định nghĩa của mô hình (`ASRModel` hoặc `ChunkFormerModel`), có thể dòng:

```python
self.ctc = SomeCTCHead(...)
```

được khai báo sau nhưng **không thêm `self.ctc` vào cấu trúc `self.encoder + self.ctc`**, tức là:

* `model` = `self.encoder`
* `self.ctc` chỉ được dùng riêng khi tính loss / decode.

---

#### 2. Bạn đang dump `model.encoder`, nhưng `model` cũng chính là `encoder`

Tức là:

```python
model, char_dict = init(...)  # model = encoder-only
```

→ Bạn chưa thực sự load `ASRModel`, mà chỉ load `encoder`.

---

### ✅ Cách kiểm tra nhanh:

Thêm đoạn sau vào cuối script:

```python
print(f"Model class: {model.__class__.__name__}")
print(f"Has encoder? {'encoder' in dict(model.named_children())}")
print(f"Has ctc? {'ctc' in dict(model.named_children())}")
```

Nếu kết quả:

```
Model class: ChunkFormerEncoder
Has encoder? False
Has ctc? False
```

→ Thì `model` chỉ là encoder.

---

### ✅ Cách sửa:

Trong `init(...)`, thay vì chỉ load encoder, bạn cần trả về full model:

```python
# Thay vì
model = ChunkFormerEncoder(...)
# Dùng
model = ASRModel(...)  # hoặc tên class đầy đủ nếu có decoder, encoder, ctc
```

---

### ✅ Bonus: Kiểm tra thủ công

Bạn cũng có thể thử:

```python
print(model)
```

Nếu chỉ hiện `ChunkFormerEncoder(...)`, thì đúng là bạn đang dump encoder mà tưởng là full model.

---

📌 **Tóm lại**: `model` của bạn hiện tại chỉ là **encoder**, vì vậy tổng số param bằng `encoder` là hợp lý. Muốn có số liệu chính xác của toàn bộ mô hình, bạn cần load đúng full `ASRModel`. Mình có thể giúp sửa `init()` nếu bạn paste định nghĩa của nó.


----------------------------------

Có, **CTC decoder** (tức là CTC head dùng để sinh logits) **có tham số**, nhưng cần phân biệt rõ 2 phần:

---

### ✅ 1. **CTC Head (Logit projection layer)** — **CÓ THAM SỐ**

Trong mô hình như bạn đang dùng:

```python
self.ctc_lo = nn.Linear(encoder_dim, vocab_size)
```

* Đây là **CTC projection layer**, chuyển từ hidden state (`[B, T, D]`) → logits (`[B, T, V]`)
* Tham số:

  * Weight: `[vocab_size, encoder_dim]`
  * Bias: `[vocab_size]`
  * Tổng: `(vocab_size × encoder_dim) + vocab_size`

📌 Ví dụ: với `vocab_size = 6992`, `encoder_dim = 512`
→ Tổng tham số = `6992 × 512 + 6992 = 3,586,896`
→ Khớp với số bạn thấy ở trên ✅

---

### ✅ 2. **CTC Decoding (Greedy / Beam Search)** — **KHÔNG CÓ THAM SỐ**

Ví dụ:

```python
log_probs = self.ctc.log_softmax(xs)   # [B, T, V]
top1 = log_probs.argmax(dim=-1)        # Greedy
```

* Đây là **phần suy luận không có tham số học**
* Nếu dùng **beam search**, có thêm cấu trúc dữ liệu và thuật toán, nhưng **không có trainable parameter**.

---

### ✅ Tổng kết:

| Thành phần           | Có tham số? | Vai trò                                        |
| -------------------- | ----------- | ---------------------------------------------- |
| `nn.Linear` CTC head | ✅ Có        | Dự đoán xác suất trên vocabulary               |
| CTC loss             | ❌ Không     | Hàm loss, không có tham số                     |
| Greedy decoder       | ❌ Không     | Duyệt chọn label đơn giản                      |
| Beam search decoder  | ❌ Không     | Có thuật toán và cây tìm kiếm, nhưng không học |

---

Nếu bạn dùng đoạn `model.ctc` thì gần như chắc chắn `model.ctc.ctc_lo` là một `nn.Linear`, và **có tham số** như đã liệt kê.
