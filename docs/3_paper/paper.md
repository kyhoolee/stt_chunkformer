**Tóm tắt ChunkFormer (tiếng Việt)**

ChunkFormer là một mô hình ASR (Automatic Speech Recognition) tối ưu cho **long-form transcription**, giải quyết hai vấn đề chính của Conformer truyền thống:

1. **Xử lý audio dài**

   * Conformer lớn chỉ chạy được 15 phút trên GPU 80 GB, trong khi ChunkFormer cho phép xử lý lên đến **16 giờ** audio trên cùng cấu hình.
   * Nhờ cơ chế **chunk-wise processing** với **relative right context**, mô hình duy trì hiệu quả và liên tục qua các phân đoạn dài.

2. **Tối ưu tài nguyên & tốc độ**

   * Loại bỏ padding phiền toái trong standard batching, ChunkFormer sử dụng **masked batching** để gom các chunk có độ dài tương đồng, giảm thiểu memory footprint và thời gian chạy hơn **3×** khi xử lý theo batch.
   * Kết quả: tiết kiệm đáng kể chi phí GPU cho hệ thống ASR thực tế.

3. **Cải thiện chất lượng**

   * Với audio dài, ChunkFormer đạt giảm **WER** tuyệt đối đến **7.7%** so với FastConformer (state-of-the-art trước đó).
   * Giữ vững độ chính xác trên các nhiệm vụ ngắn, tương đương hoặc tốt hơn Conformer gốc.

---

*ChunkFormer kết hợp hiệu năng cao của Conformer với cơ chế chunk streaming thông minh, mang lại khả năng mở rộng (scalability) và tiết kiệm chi phí cho các ứng dụng ASR công nghiệp.*


---

Introduction

Trong bối cảnh ASR công nghiệp quy mô lớn, việc xử lý chuỗi audio dài gặp 3 thách thức chính:

    - Tiêu thụ tài nguyên GPU: mô hình attention-based như Conformer tốn bộ nhớ và thời gian tính toán O(L²d) theo độ dài L, dẫn đến chi phí hàng triệu đô la khi phục vụ hàng triệu người dùng.

    - Hiệu năng trên audio dài: các mô hình hiện tại chỉ xử lý được tối đa vài chục phút (Conformer ~15 phút), trong khi nhu cầu long-form transcription (giờ đến chục giờ) vẫn chưa được đáp ứng.

    - Batching không hiệu quả: độ dài đầu vào thay đổi lớn buộc padding, khiến tốc độ và bộ nhớ tiêu tốn tăng đột biến (ví dụ batch gồm audio 1 giờ và 1 giây vẫn pad thành hai audio 1 giờ).

Các phương pháp hiện có:

    - Conformer [1]: 
        - Key-idea:
            - kết hợp convolutional local features + self-attention, 
        - O(L²d) giới hạn tối đa độ dài.

    - Efficient-Conformer [2]: 
        - Key-idea: 
            - tăng downsampling 8× với stride-2 depthwise conv, 
            - grouped attention 
        - giảm độ phức tạp xuống O(L²d/g).

    - FastConformer [3]: 
        - Key-idea: 
            - downsampling 8× ngay đầu, 
            - limited-context attention (LongFormer) 
        - giảm thành O(L×(w×2+1)×d) với w=128.

    - Segmentation + batching: 
        - Key-idea:
            - cắt audio cố định hoặc theo VAD, 
            - giữ context qua buffer hoặc merge segments (WhisperX [5]),
        - Issue mất liên tục context và vẫn chịu padding.

Chú thích 
    - L: độ dài chuỗi đầu vào (số frames hoặc tokens).
    - d: kích thước embedding / hidden dimension của mô hình.
    - g: (Efficient-Conformer)
        - số nhóm (group size) trong grouped attention. 
        - Chia attention heads hoặc keys thành g nhóm để tính song song.
    - w: (FastConformer)
        - window size trong limited-context attention. 
        - Mỗi token chỉ attention tới w token trước và w token sau, 
            - nên tổng window là (w·2 + 1).


---------------

**Tóm tắt ý tưởng tổng quan của ChunkFormer**

Trong bối cảnh **Introduction** đã nêu, ChunkFormer giải quyết hai thách thức chính
    - xử lý audio dài 
    - batching không padding—bằng 

Với hai cơ chế then chốt:

1. **Endless Decoding (Chunk-Based Streaming với Relative Right Context)**

   * Mỗi chunk (ví dụ 80 ms) 
    - được xử lý tuần tự qua các layer như Conformer, 
    - nhưng vẫn “nhìn” được sang **right context** của chunk kế tiếp 
        - thông qua các bias/relative-position embedding.
   * Cơ chế này cho phép 
    - duy trì **liên tục context** giữa các chunk 
    - mà không phải tải toàn bộ audio lên GPU trước, 
    - nhờ đó mở rộng khả năng xử lý lên đến hàng chục giờ trên GPU 80 GB.
   * Về mặt tính toán, 
    - attention vẫn giới hạn trong window nhỏ xung quanh từng token, 
    - giữ complexity ≈ O(L · d) thay vì O(L² · d).

2. **Masked Batch Decoding (Batching Không Padding)**

   * Thay vì standard batching phải pad tất cả chunk về độ dài tối đa → lãng phí bộ nhớ và thời gian, 
    - dùng **mask** để chỉ tính toán trên vị trí thực sự có dữ liệu.
   * Với kỹ thuật này, 
    - batch gồm nhiều audio có độ dài khác nhau (ví dụ 1 giờ và 1 giây) 
    - sẽ tiêu tốn tài nguyên tương đương batch chỉ có một audio 1 giờ và 1 giây, 
    - thay vì tương đương hai audio 1 giờ.
   * Kết quả là 
    - giảm > 3× memory footprint và execution time trong batch processing.

3. **Kết quả**

   * **Khả năng mở rộng (scalability)**: 
    - xử lý tới 16 giờ audio trên GPU 80 GB (gấp 1.5× so với FastConformer).
   * **Chất lượng**: 
    - giảm **WER** đến 7.7% trên long-form datasets, 
    - đồng thời giữ vững hoặc cải thiện độ chính xác cho audio ngắn.
   * **Tài nguyên & tốc độ**: 
    - tiết kiệm đáng kể chi phí GPU, 
    - tối ưu cho hệ thống ASR (production).

> **Kết luận**: ChunkFormer kết hợp 
    - chunk-wise streaming với contextual bias 
    - masked batching để đạt được khả năng xử lý audio dài liên tục, 
    - giảm padding overhead, 
    - giữ performance cao mà không yêu cầu GPU bộ nhớ lớn.

----
# ChunkFormer Training Core Logic
# ChunkFormer Training Core Logic

## A. Chunk-wise Streaming với Contextual Bias – Chi tiết biến & luồng tính toán

### Motivation & Intuition

* **Vấn đề**: Xử lý audio dài liên tục yêu cầu giữ context giữa các phần (chunks) mà không thể tải toàn bộ lên GPU.
* **Giải pháp**: Dùng **chunk-wise streaming** để chạy song song và **relative positional bias** để mô hình biết khoảng cách giữa frames dù mỗi batch chỉ chứa chunk ngắn.
* **Lợi ích**:

  * **Memory-efficient**: không cần toàn bộ sequence.
  * **Parallelism**: batch-axis xử lý nhiều chunk cùng lúc.
  * **Consistency**: training match inference endless decoding.

---

### 1. Các tham số & định nghĩa

---

### 1. Các tham số & định nghĩa

* **c**: chunk size (số frames mỗi chunk).
* **r**: right-context size (số frames tương lai mỗi chunk cần “nhìn” sang).
* **N**: số encoder layers.
* **m**: số chunk chính trong 1 batch (batch-axis).
* **r_rel**: tổng số future frames cần để phục vụ toàn bộ N layers:

  ```
  r_rel = r + max(c, r) * (N - 1)
  ```
* **X**: toàn bộ sequence frames độ dài L.
* **X_i**: chunk thứ i, chứa frames:

  ```
  X_i = [ X[i*c], X[i*c + 1], …, X[i*c + (c - 1)] ]
  ```
* **X_future**: future chunk chứa frames:

  ```
  X_future = X[m*c : m*c + r_rel]
  ```
* **W_q, W_k, W_v**: ma trận weight cho query/key/value.
* **W_R**: ma trận weight cho relative positional encoding.
* **R_d**: positional embedding cho khoảng cách d = j – t.
* **u, v**: bias vectors học được.
* **d_k**: hidden dimension per attention head.
* **β**: hệ số scale trong softmax.

---

### 2. Chuẩn bị batch input

1. Lấy m chunk chính:

   ```
   [ X_0, X_1, …, X_{m-1} ]
   ```
2. Tính `r_rel` theo công thức trên.
3. Lấy future frames:

   ```
   X_future = X[m*c : m*c + r_rel]
   ```
4. Tạo batch input gồm m+1 chunks:

   ```
   X_batch = [ X_0, X_1, …, X_{m-1}, X_future ]
   ```

---

### 3. Luồng tính toán attention cho mỗi chunk X_i

Với mỗi query frame index j ∈ [i*c, i*c + c – 1] trong chunk X_i:

1. Xác định window key:
   t_start = i*c – l (left cache)
   t_end   = i*c + (c – 1) + r (right context)
   t ∈ [t_start, t_end]
2. Tính energy e_{j,t} cho mọi t trong window:

   ```
   e_{j,t} = (
     x_j W_q · (x_t W_k)^T
     + x_j W_q · (R_{j-t} W_R)^T
     + u · (x_t W_k)^T
     + v · (R_{j-t} W_R)^T
   ) / sqrt(d_k)
   ```
3. Bình thường hóa softmax:

   ```
   α_{j,t} = exp(β·e_{j,t})
             / Σ_{t=t_start..t_end} exp(β·e_{j,t})
   ```
4. Tính output z_j:

   ```
   z_j = Σ_{t=t_start..t_end} α_{j,t} · (x_t W_v)
   ```

---

### 4. Luồng tổng quát cho batch

1. **Input**: X_batch shape = (batch_size=m, seq_len=c + extra)
2. **Encoder Layer 1**: convolution & attention sử dụng cache left và X_future
3. **Lặp** cho tới layer N, luôn có X_future để cung cấp context
4. **Output**: lấy chỉ X_0…X_{m-1} từ layer N
5. **CTC/AED head**: tính logits và loss

---

### 5. Kết quả & lợi ích

* **Endless decoding**: training khớp inference nhờ X_future
* **Parallelism**: batch-axis xử lý m chunk đồng thời.
* **Efficiency**: giới hạn attention window, giảm complexity so với O(L²·d).

---

## Data Flow (text diagram)

```
              +------------------------------------+
              |   Concatenate chunks + X_future   |
              |           X_batch                 |
              +----------------+-------------------+
                               |
                               v
              +------------------------------------+
              |     Encoder Layer 1 (Conv+Attn)   |
              +----------------+-------------------+
                               |
                               v
              [ ... lặp tới Layer N ... ]
                               |
                               v
              +------------------------------------+
              |    Select outputs X_0..X_{m-1}    |
              +----------------+-------------------+
                               |
                               v
              +------------------------------------+
              |       CTC/AED Head & Loss         |
              +------------------------------------+
```

*Each arrow represents tensor flow; attention modules use relative positional bias and caches.*

-- Luồng tính toán chi tiết 

Step 0: Waveform → Fbank (overlapping windows)
────────────────────────────────────────────────
Original waveform (length L_samples)
  ↓ Split into overlapping frames:
    • Window = 25 ms (e.g. 400 samples)  
    • Hop    = 10 ms (e.g. 160 samples)  
  → Produces T_total frames of size M (e.g. 80)  
Feature sequence: shape = (T_total, M)

Step 1: Split feature sequence into disjoint chunks
────────────────────────────────────────────────
Let T_chunk = number of frames per chunk  
Chunks:
  Chunk 0_feats = frames [0 … T_chunk–1]  
  Chunk 1_feats = frames [T_chunk … 2·T_chunk–1]  
  …  
  Chunk m–1_feats = frames [(m–1)·T_chunk … m·T_chunk–1]  
Remaining frames → used to build X_future_feats

Step 2: Build batch with future context
────────────────────────────────────────────────
Compute r_rel = r + max(c, r)·(N–1)  
Extract future frames:
  X_future_feats = frames [m·T_chunk … m·T_chunk + r_rel – 1]  
Batch input:
  X_batch_feats = [ Chunk 0_feats,
                    Chunk 1_feats,
                    …,
                    Chunk m–1_feats,
                    X_future_feats ]

Step 3: Encoder layers (1→N)
────────────────────────────────────────────────
For i in 1…N:
  • Convolution module (uses left-cache + right-context)
  • Attention module (windowed + relative bias)
Input & output shape each layer = (batch_size=m, seq_len=T_chunk + extra, D)

Step 4: Extract chunk outputs
────────────────────────────────────────────────
From layer N output, drop the part corresponding to X_future_feats  
Keep only the first T_chunk frames for each of the m chunks

Step 5: CTC/AED head & decode
────────────────────────────────────────────────
• CTC: concatenate logits of all chunks → streaming CTC decode  
• AED: feed decoder state across chunks → streaming autoregressive decode

Step 6: Final transcript
────────────────────────────────────────────────
• CTC: remove blanks & collapse repeats → join chunk transcripts  
• AED: concatenate generated tokens (no duplication)  


-------------




## C. ChunkFormer Convolution

### Motivation & Intuition

* **Vấn đề**: Convolution module trong FastConformer có receptive field nhỏ sau subsampling, dẫn đến hiệu năng kém hơn Conformer full-context.
* **Mục tiêu**: Mở rộng receptive field và ổn định training cho mô hình lớn, đồng thời giữ memory-efficient cho audio dài.

### 1. Subsampling & Kernel Size

* **Downsampling**: sử dụng subsample module từ FastConformer, tăng số filters lên **D** (embedding dimension của encoder).
* **Kernel size**: tăng kernel size từ mặc định lên **15** để receptive field tương đương Conformer sau downsampling.
* **LayerNorm**: thay BatchNorm bằng LayerNorm trong convolution modules để ổn định gradient khi training dài.

### 2. Limited Context & Masked Batching

* Với audio dài, đầu vào subsample convolution vẫn lớn → memory bottleneck.
* Thay vì split batch thủ công (theo channel hoặc batch size), ChunkFormer:

  1. **Split audio** thành chunks nhỏ (size c) → batch-axis.
  2. **Masked batching** cho subsample: process mỗi chunk độc lập, tận dụng parallelism của framework.

### 3. Luồng xử lý Convolution

```
Input wave_feats: shape (batch_size, T, M)
  ↓ Subsample layer (Depthwise conv stride=2, filters=D)
  ↓ ConvModule (kernel_size=15, LayerNorm)
Output conv_feats: shape (batch_size, T/2, D)
```

* **Batch splitting**: audio dài được chuyển thành chunks, mỗi chunk được subsample + conv độc lập.
* **Masked batching** đảm bảo không xử lý vùng padding hay overlap.

### 4. Lợi ích

* **Receptive field** tương đương Conformer sau downsampling.
* **Stability**: LayerNorm giúp training large model lâu dài.
* **Scalability**: subsample + chunk processing cho phép xử lý audio dài trên GPU ít memory.

---

## D. Experiments

### 1. Datasets

**Small-scale (public)**

* **Train**: 960 h Librispeech (LS)
* **Test [SM]**: 11 h LS test set
* **Test [L]**:
  * 3 h Tedlium-v3
  * 39 h Earnings-21
  * 119 h Earnings-22

**Large-scale (internal Vietnamese)**

* **Train**: 25 000 h multi-domain (Reading, Conversation, Telephony, YouTube)
* **Test [SM]**:
  * 2.8 h Reading
  * 11 h Conversation
  * 7.5 h Telephony
* **Test [L]**: 3.2 h YouTube

*SM = short–medium form; L = long form.*

### 2. Model & Loss

* **Architecture**: CTC–AED hybrid
* **Encoder**: 17 layers, 8 heads, dim=512
* **Loss**: L_total = 0.3 L_CTC + 0.7 L_AED
* **Inference**: CTC decoder + greedy search

### 3. Training Setup

* **Features**: 80‐dim filter-bank (25 ms/10 ms)
* **Augmentation**: Speed-Perturb, SpecAugment
* **Vocab**: BPE 5 000 tokens, embed=256
* **Hardware**: 8×H100 GPUs, mixed precision
* **Schedule (full-context)**:
  * Small: 200 epochs (400 k steps), peak LR = 1e-3 (warm-up 15 k steps)
  * Large: fine-tune 50 epochs (100 k steps), peak LR = 1e-5
* **Optimizer**: Adam + Noam warm-up
* **Checkpointing**: average last 50 (small) / 10 (large)
* **Dynamic Context Training**: vary (l_att, c, r) for robustness


**III-B. Kết quả chính**

---

### 1. Khả năng xử lý audio dài (Table III)

* **Conformer (115 M)**: tối đa 15 phút
* **FastConformer (109 M, [128,1,128])**: tối đa 675 phút
* **ChunkFormer (110 M)**

  * [256,128,128] → 760 phút
  * [128,64,128]  → **980 phút**

> ChunkFormer với cấu hình [128,64,128] cho phép xử lý **gần 16 giờ** audio trên GPU 80 GB – gấp ~65× so với Conformer và ~1.5× so với FastConformer.

---

### 2. Độ chính xác (WER)

* **Full-context**: ChunkFormer đạt WER trung bình 18.36%, tương đương Conformer/Efficient/Squeezeformer.
* **Limited-context [128,64,128]**: ChunkFormer dẫn đầu trên Tedlium-v3/Earnings-21/Earnings-22 với WER 21.32%, 31.73%, 48.56% (giảm tuyệt đối 7.7% so với Conformer trên Earnings-21).
* **Large-scale Vietnamese**: WER 5.05% (Reading), 7.17% (Conversation), 11.85% (Telephony), 14.05% (YouTube).

> ChunkFormer cải thiện rõ rệt trên long-form nhờ cơ chế chunk-wise streaming và giữ vững hiệu năng cho short-medium form.

---

### 3. Hiệu năng & tài nguyên (Table IV)

| Model                       | Memory (GB) | Time (s) | FLOPS (T) | (Encoder-only) Mem | Time    | FLOPS    |
| --------------------------- | ----------- | -------- | --------- | ------------------ | ------- | -------- |
| FastConformer (naive batch) | 73.4        | 2.4      | 64.3      | 26.4               | 2.1     | 61.5     |
| ChunkFormer (naive batch)   | 34.5        | 2.7      | 65.2      | 25.8               | 2.2     | 56.7     |
| ChunkFormer + Masked batch  | **19.6**    | **0.8**  | **19.3**  | **8.1**            | **0.7** | **16.8** |

* **Masked batching** giảm 1.5–3.7× memory, 3–3.4× time, 1.8–3.7× FLOPS so với naive batch.
* Khi so encoder-only, ChunkFormer + Masked batch vẫn **nhẹ nhất** về memory/time/FLOPS.

> Masked batch ChunkFormer tận dụng tối đa parallelism của framework, tối ưu bộ nhớ và tốc độ trên mọi độ dài audio.

---

**Kết luận**: ChunkFormer không chỉ **mở rộng** đáng kể khả năng xử lý audio dài, mà còn **cải thiện WER** và **tiết kiệm tài nguyên** nhờ cơ chế chunk-wise streaming, limited-context attention và masked batching.
