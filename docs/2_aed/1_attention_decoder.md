Tuyệt! Dựa trên các thông tin bạn cung cấp, mình có thể **dựng lại logic `TransformerDecoder`** (AED head) từ `left_decoder` với đầy đủ thông số như sau:

---

## ✅ Phân tích kiến trúc

Từ checkpoint:

* `embed.0.weight → (6992, 512)` → **vocab\_size = 6992**, `d_model = 512`
* `feed_forward.w_1.weight → (2048, 512)` → **ffn\_dim = 2048**
* Có 3 block: `decoders.0`, `decoders.1`, `decoders.2` → **num\_blocks = 3**
* Tất cả attention layers đều có `linear_q.weight → (512, 512)` → suy ra: **attention\_heads = 8** (mỗi head 64 dim)

---

## 🧩 Mô hình `TransformerDecoder` tương thích với `left_decoder`

Bạn có thể đặt vào `chunkformer_vpb/modules/transformer_decoder.py`:

```python
import torch
import torch.nn as nn
from wenet.transformer.attention import MultiHeadedAttention
from wenet.transformer.decoder_layer import DecoderLayer
from wenet.transformer.positionwise_feed_forward import PositionwiseFeedForward
from wenet.utils.common import make_pad_mask


class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size, encoder_output_size, attention_heads=8,
                 linear_units=2048, num_blocks=3, dropout_rate=0.1):
        super().__init__()

        self.embed = nn.Sequential(
            nn.Embedding(vocab_size, encoder_output_size),
            nn.Dropout(p=dropout_rate)
        )

        self.decoders = nn.ModuleList([
            DecoderLayer(
                size=encoder_output_size,
                self_attn=MultiHeadedAttention(attention_heads, encoder_output_size, dropout_rate),
                src_attn=MultiHeadedAttention(attention_heads, encoder_output_size, dropout_rate),
                feed_forward=PositionwiseFeedForward(
                    encoder_output_size, linear_units, dropout_rate
                ),
                dropout_rate=dropout_rate
            ) for _ in range(num_blocks)
        ])

        self.after_norm = nn.LayerNorm(encoder_output_size)
        self.output_layer = nn.Linear(encoder_output_size, vocab_size)

    def forward(self, tgt, tgt_mask, memory, memory_mask):
        xs = self.embed(tgt)
        for decoder in self.decoders:
            xs, tgt_mask, memory, memory_mask = decoder(xs, tgt_mask, memory, memory_mask)
        xs = self.after_norm(xs)
        return self.output_layer(xs)
```

---

## 📥 Cách gọi trong inference

Bạn có thể thêm vào `asr_model.py`:

```python
from chunkformer_vpb.modules.transformer_decoder import TransformerDecoder

self.decoder = TransformerDecoder(
    vocab_size=6992,
    encoder_output_size=512,
    attention_heads=8,
    linear_units=2048,
    num_blocks=3,
    dropout_rate=0.1
)
```

---

## 🧪 Để kiểm tra hoạt động:

Trong `decode.py` hoặc script test:

```python
tgt = torch.tensor([[1, 23, 45, 2]])  # [B, T] (with <sos> and <eos>)
tgt_mask = (tgt != 0).unsqueeze(1)  # padding mask
encoder_out = ...  # [B, T, D]
encoder_mask = ...  # [B, 1, T]

logits = model.decoder(tgt, tgt_mask, encoder_out, encoder_mask)
```

---

## 📌 Ghi chú

* Nếu bạn muốn dùng `attention_rescoring` hoặc `beam_search`, mình có thể viết thêm hàm đó.
* Cần giữ đúng tên các module (`embed`, `after_norm`, `output_layer`, `decoders.*`) để `load_state_dict` không bị mismatch.

---

Bạn muốn mình viết thêm đoạn `attention_rescoring()` inference sample cho `ChunkFormer` với AED không?
