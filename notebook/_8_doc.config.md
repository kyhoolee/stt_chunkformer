Dưới đây mình bóc tách và giải thích từng phần chính trong file `train_u2++_chunkformer_small.yaml` của WeNet, để bạn nắm rõ ý nghĩa và dễ tùy chỉnh cho fine-tune:

---

## 1. Network Architecture

```yaml
encoder: chunkformer
encoder_conf:
    output_size: 256            # embedding/hidden size của attention
    attention_heads: 4          # số head trong multi-head attention
    linear_units: 2048          # hidden size của feed-forward layer
    num_blocks: 12              # số layer encoder
    dropout_rate: 0.1
    positional_dropout_rate: 0.1
    attention_dropout_rate: 0.1
    input_layer: dw_striding    # type của input layer (conv2d với stride)
    normalize_before: true      # pre-norm vs post-norm
    cnn_module_kernel: 15       # kernel size cho 1D convolution
    use_cnn_module: True        # bật conv-module trong chunkformer
    activation_type: 'swish'    # activation cho conv-module
    pos_enc_layer_type: 'chunk_rel_pos'
    selfattention_layer_type: 'chunk_rel_seflattn'
    # Nếu muốn train “dynamic context” (ghép full + chunk), bật các dòng bên dưới:
    # dynamic_conv: true
    # dynamic_chunk_sizes: [-1, -1, 64, 128, 256]
    # dynamic_left_context_sizes: [64, 128, 256]
    # dynamic_right_context_sizes: [64, 128, 256]
```

* **chunk\_rel\_…**: dùng relative positional encoding và relative self-attention chuyên cho chunked input.
* **dynamic\_…**: cho phép random chọn giữa full-context (`-1`) và các cấu hình chunked trong mỗi batch.

```yaml
decoder: bitransformer
decoder_conf:
    attention_heads: 4
    linear_units: 2048
    num_blocks: 3
    r_num_blocks: 3            # số layer của right-to-left decoder (bidirectional)
    dropout_rate: 0.1
    positional_dropout_rate: 0.1
    self_attention_dropout_rate: 0.1
    src_attention_dropout_rate: 0.1
```

* `bitransformer`: decoder hai chiều (forward + reverse) cho attention-based decoding.

---

## 2. Tokenizer & CTC

```yaml
tokenizer: bpe
tokenizer_conf:
  symbol_table_path: 'data/lang_char/train_960_bpe5000_units.txt'
  bpe_path:          'data/lang_char/train_960_bpe5000.model'
  split_with_space:  false
  special_tokens:
    <blank>: 0
    <unk>:   1
    <sos>:   2
    <eos>:   2
```

* Dùng BPE 5 000 token, với `<blank>` ID=0 cho CTC.

```yaml
ctc: ctc
ctc_conf:
  ctc_blank_id: 0
```

---

## 3. Input Features & CMVN

```yaml
cmvn: global_cmvn
cmvn_conf:
  cmvn_file:    'data/train_960/global_cmvn'
  is_json_cmvn: true

dataset: asr
dataset_conf:
    # filter audio dài/short
    filter_conf:
        max_length: 40960    # tối đa frames sau fbank (≈40960×10 ms = 409.6 s)
    resample_conf:
        resample_rate: 16000
    speed_perturb: true      # bật Speed-Perturb (0.9, 1.0, 1.1)
    fbank_conf:
        num_mel_bins: 80
        frame_shift: 10
        frame_length: 25
        dither: 1.0
    spec_aug: true           # bật SpecAugment
    spec_aug_conf:
        num_t_mask: 2
        num_f_mask: 2
        max_t: 50             # che ≤50 frames
        max_f: 10             # che ≤10 bins
    spec_sub: false          # SpecSub deprecated
    shuffle: true
    sort:    false
    batch_conf:
        batch_type: 'dynamic' # dynamic = gom batch theo max_frames_in_batch
        max_frames_in_batch: 120000
        pad_feat: True        # True để kích hoạt masked-batch trong decode/train
```

* **dynamic batch**: tự động gom càng nhiều sample sao cho tổng frames ≤120 000.
* **pad\_feat=True**: bật zero-padding + mask để enable masked-batch processing.

---

## 4. Model-level & Hybrid Loss

```yaml
model: asr_model
model_conf:
    ctc_weight: 0.3           # λ cho CTC loss
    lsm_weight: 0.1           # label smoothing trong AED
    reverse_weight: 0.3       # trọng số của reverse decoder
```

* Kết hợp **CTC + AED** với λ=0.3 căn bản như paper.

---

## 5. Training & Optimization

```yaml
grad_clip: 5                # clip gradient norm
accum_grad: 4               # gradient accumulation steps
max_epoch: 200
log_interval: 100

optim: adamw
optim_conf:
    lr: 0.001               # 1e-3 peak LR cho pre-train small
    weight_decay: 0       # (nếu khai báo ở trên)

scheduler: warmuplr         # linear warm-up + decay
scheduler_conf:
    warmup_steps: 25000     # số step warm-up
```

* **accum\_grad=4**: chia batch lớn ra làm 4 micro-batches, tiết kiệm GPU mem.
* Pre-train small dùng LR=1e-3, warm-up 25 000 steps.

---

### Áp cho Fine-tune Call-Bot

* **Chỉ cần** giữ lại cấu trúc encoder/decoder, model\_conf (ctc\_weight=0.3), dataset\_conf rồi giảm:

  * `optim_conf.lr` → 5e-5
  * `scheduler_conf.warmup_steps` → 2 000
  * `max_epoch` → 10–20
  * `batch_conf.pad_feat` vẫn `True` để dùng masked-batch.
* Tắt Speed-Perturb nếu không cần, hoặc giảm spec\_aug\_conf về phù hợp audio call-bot.

---

Hy vọng phần giải thích này giúp bạn hiểu rõ từng block trong config và dễ điều chỉnh cho fine-tune!
