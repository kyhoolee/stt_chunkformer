Model bạn đang dùng là một mô hình **ASR (Automatic Speech Recognition)** hiện đại có cấu trúc **Chunk-Streaming Conformer Encoder + CTC (Connectionist Temporal Classification)**. Cụ thể hơn, đây là một phiên bản nâng cấp gọi là **ChunkFormer**, được thiết kế để hoạt động hiệu quả **trong chế độ streaming** (phát hiện và giải mã theo từng khối dữ liệu âm thanh) nhưng vẫn giữ được chất lượng nhận dạng như mô hình full context.

---

## ✅ Tổng quan kiến trúc

```text
[Input audio (waveform)] 
     ↓
[Feature Extraction (FBank: 80-dim, frame 25ms, stride 10ms)]
     ↓
[DepthwiseConvSubsampling (x3 Conv2D → Linear projection)]
     ↓
[17 x ChunkFormerEncoderLayer (Multi-Head Attention + FF + Conv)]
     ↓
[CTC Linear + log_softmax]
     ↓
[CTC Decoder (Greedy / Beam)]
     ↓
[Text output]
```

---

## 🔍 Thành phần chính

### 1. **GlobalCMVN**

* Chuẩn hóa mean-variance toàn bộ đặc trưng đầu vào.
* Giúp model ổn định hơn, giảm ảnh hưởng từ biên độ âm thanh đầu vào.

---

### 2. **DepthwiseConvSubsampling**

* 3 tầng `Conv2D` liên tiếp với stride (2,2) → giảm chiều thời gian 8 lần.
* Sau đó là `Linear` để biến đổi thành embedding 512 chiều.
* Đây là **subsampling** vừa giúp giảm thời gian tính toán, vừa trích xuất đặc trưng mạnh hơn nhờ tích chập.

---

### 3. **Positional Encoding (StreamingRelPositionalEncoding)**

* Dùng kiểu **relative positional encoding** để hỗ trợ tốt hơn cho chunk-based và streaming.
* Có thể generalize trong các chunk khác nhau mà không phụ thuộc tuyệt đối vào vị trí tuyệt đối.

---

### 4. **ChunkFormerEncoderLayer × 17**

* Mỗi layer bao gồm:

  * **Macaron FFN** (gồm 2 FFN → giúp biểu diễn tốt hơn).
  * **StreamingRelPositionMultiHeadedAttention**: attention theo vị trí tương đối.
  * **Depthwise Separable Conv1D Module** (Conv Module): bắt các đặc trưng local.
  * **LayerNorm & Dropout** sau mỗi block.
* Hỗ trợ **caching** attention và conv state → phục vụ streaming inference.

---

### 5. **CTC Head**

* Lớp `Linear(512 → vocab_size)` + `CTCLoss`.
* Cho phép mô hình huấn luyện không cần alignment.
* Dễ kết hợp với các mô hình decoder khác sau này.

---

## 🧠 Đặc điểm nổi bật

| Đặc điểm                    | Ý nghĩa                                                                |
| --------------------------- | ---------------------------------------------------------------------- |
| **Streaming bằng Chunk**    | Giúp model xử lý real-time, tiết kiệm RAM, latency thấp                |
| **Subsampling x8**          | Giảm số lượng frame, tăng tốc nhưng vẫn giữ thông tin quan trọng       |
| **Conformer Layer mạnh mẽ** | Kết hợp Attention (global) + Conv (local) → rất mạnh cho ASR           |
| **CTC output**              | Dễ huấn luyện, inference nhanh (greedy hoặc beam search)               |
| **Attention/Conv cache**    | Hỗ trợ mô hình nhớ context trước mà không cần nhìn toàn bộ tín hiệu    |
| **Masking logic phức tạp**  | Quản lý context trái/phải phù hợp theo từng chunk để đảm bảo chính xác |

---

## 🧪 Đánh giá

| Tiêu chí                     | Nhận xét                                                             |
| ---------------------------- | -------------------------------------------------------------------- |
| **Độ chính xác**             | Cao nếu cấu hình context hợp lý, vocab đủ tốt. Có thể đạt gần SOTA.  |
| **Tốc độ decode**            | Nhanh do dùng greedy decode CTC và đã giảm frame với subsampling.    |
| **Tính mở rộng**             | Tốt: dễ kết hợp thêm RNN/Transformer decoder, LM, rescoring...       |
| **Tính ổn định**             | Có CMVN + dropout + LayerNorm đầy đủ → mô hình rất ổn định           |
| **Khả năng học ngữ cảnh xa** | Có nhưng không bằng full-seq, vì giới hạn chunk size + right context |

---

## 📦 Khi nào nên dùng mô hình này?

* ✅ ASR tiếng Việt real-time latency thấp (<500ms)
* ✅ Thiết bị hạn chế tài nguyên (RAM/VRAM)
* ✅ Yêu cầu inference nhanh và model nhỏ (<150MB)
* ✅ Có thể fine-tune cho domain cụ thể (ngân hàng, chăm sóc khách hàng…)

---

## 📘 Gợi ý tiếp theo cho bạn

* Bạn có thể:

  * Gắn thêm **CTC Prefix Beam Search** để decode tốt hơn.
  * Fine-tune vocab/loss nếu nhận dạng sai nhiều từ hiếm.
  * Huấn luyện thêm với dữ liệu riêng để tăng WER.

---

Dưới đây là sơ đồ dạng **text** minh họa kiến trúc mô hình **ChunkFormer ASR** bạn đang dùng, bao gồm input → encoder → output với **kích thước tensor (dimension)** rõ ràng từng bước:

---

## 🔷 **📥 Input pipeline**

```text
[Audio waveform] (B=1, T=~38000 samples @16kHz)
    ↓
[FBank features (frame_length=25ms, frame_shift=10ms)]
    → Output: [B=1, T=236, D=80]
```

---

## 🔶 **🧠 ChunkFormer Encoder**

### 0. 📏 **GlobalCMVN**

```text
Input:  [B=1, T=236, D=80]
Output: [B=1, T=236, D=80]   (chuẩn hóa mean/var)
```

---

### 1. 🧱 **DepthwiseConvSubsampling**

Chuỗi 3 conv2d + linear:

* Conv2D(1, 512, kernel=(3,3), stride=(2,2))
* Depthwise Conv2D (group=512), stride=(2,2)
* Conv2D(512,512), stride=(2,2)
* Cuối cùng flatten + Linear(4608 → 512)

```text
Input:  [B=1, T=236, D=80]   → reshaped to [B=1, C=1, T=236, D=80]
Conv+Subsampling: giảm chiều T còn ≈ T/8 → 236 → 29
Flatten: [B, T=29, D'=4608]
Linear:  [B=1, T=29, D'=4608] → [B=1, T=29, D=512]
```

---

### 2. 🎯 **StreamingRelPositionalEncoding**

```text
Input:  [B=1, T=29, D=512]
Output: [B=1, T=29, D=512] (thêm vị trí tương đối cho attention)
```

---

### 3. 🔁 **17 x ChunkFormerEncoderLayer**

#### Mỗi layer gồm:

* **Macaron Feed Forward** (x2)
* **Multi-Head Attention** (rel pos, streaming)
* **Conv Module**: Depthwise Conv1D(kernel=15)
* **LayerNorm** sau mỗi block

```text
Input:  [B=1, T=29, D=512]
Output: [B=1, T=29, D=512] (không đổi shape, chỉ biến đổi embedding)
```

#### Tổng thể sau 17 layer:

```text
Output: [B=1, T=29, D=512]
```

> Với `chunk_size=64`, sau `embed` thì chiều T ≈ 64 → giữ xuyên suốt các layer.

---

### 4. 🟦 **LayerNorm cuối cùng (after\_norm)**

```text
Input:  [B=1, T=64, D=512]
Output: [B=1, T=64, D=512]
```

---

### 5. 📤 **CTC Layer**

```text
Linear: [B=1, T=64, D=512] → [B=1, T=64, V=6992]   (vocab size)
log_softmax → lấy top-1 theo dim V

CTC Greedy decode:
→ framewise token ids: [B=1, T=64] → postprocess → [28] valid tokens
```

---

## ✅ **📝 Final Output**

```text
framewise_token_ids: [28]
→ Text decoded: "nửa vòng trái đất hơn bảy năm"
```

---

## 🧩 Summary Table

| Stage                       | Shape In       | Shape Out      | Notes                        |
| --------------------------- | -------------- | -------------- | ---------------------------- |
| Audio → FBank               | \[1, \~38000]  | \[1, 236, 80]  | frame\_length=25ms, hop=10ms |
| GlobalCMVN                  | \[1, 236, 80]  | \[1, 236, 80]  | normalize                    |
| ConvSubsampling             | \[1, 236, 80]  | \[1, 64, 512]  | T reduced by 8×              |
| PositionalEncoding          | \[1, 64, 512]  | \[1, 64, 512]  | streaming relpos             |
| 17 x EncoderLayer           | \[1, 64, 512]  | \[1, 64, 512]  | contextual processing        |
| Final LayerNorm             | \[1, 64, 512]  | \[1, 64, 512]  | output embedding             |
| Linear + log\_softmax (CTC) | \[1, 64, 512]  | \[1, 64, 6992] | vocab probs                  |
| Greedy decode               | \[1, 64, 6992] | \[28]          | (framewise token IDs)        |

---


