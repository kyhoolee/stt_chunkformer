Dưới đây là gợi ý cách bạn có thể hiện thực hàm loss cho ChunkFormer, tham khảo trực tiếp từ cách Wenet tính trong `wenet/transformer/asr_model.py`:

```python
class ChunkFormerASRModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        # … khởi tạo encoder, ctc, decoder …
        self.ctc_weight = config.model_conf['ctc_weight']  # 0.3
        self.reverse_weight = config.model_conf.get('reverse_weight', 0.0)
        self.ctc_loss_fn = nn.CTCLoss(blank=config.ctc_conf['ctc_blank_id'],
                                       zero_infinity=True)
        self.att_loss_fn = LabelSmoothingLoss(
            size=vocab_size,
            padding_idx=config.tokenizer_conf['<blank>'],
            smoothing=config.model_conf.get('lsm_weight', 0.0)
        )
    def forward(self, feats, feat_lens, ys, ys_lens):
        # 1) forward full-context (nếu dùng dynamic)
        loss_full = None
        #    … code tính encoder_out_full, mask_full …
        #    loss_ctc_full, _ = self.ctc(encoder_out_full, mask_full, ys, ys_lens)
        #    loss_att_full, _ = self._calc_att_loss(encoder_out_full, mask_full, ys, ys_lens)
        #    loss_full = self.ctc_weight * loss_ctc_full \
        #                + (1 - self.ctc_weight) * loss_att_full

        # 2) forward chunk-context
        encoder_out, mask = self.forward_chunk(feats, feat_lens)
        # a) CTC
        ctc_logits = self.ctc(encoder_out)  # [B, T, V]
        logp = ctc_logits.log_softmax(-1).permute(1,0,2)
        loss_ctc = self.ctc_loss_fn(
            logp, ys, mask.sum(-1), ys_lens
        )
        # b) AED (forward decoder + label-smoothing)
        decoder_in = ys[:, :-1]
        att_out = self.decoder(encoder_out, decoder_in, mask)
        B, L, V = att_out.size()
        loss_att = self.att_loss_fn(
            att_out.view(-1, V),
            ys[:,1:].reshape(-1)
        )
        # c) Nếu dùng reverse (bitransformer)
        if self.reverse_weight > 0:
            rev_in = ys[:, 1:]
            rev_out = self.r_decoder(encoder_out, rev_in, mask)
            loss_rev = self.att_loss_fn(
                rev_out.view(-1, V),
                ys[:,:-1].reshape(-1)
            )
            # gộp vào att loss
            loss_att = (1 - self.reverse_weight) * loss_att \
                       + self.reverse_weight * loss_rev

        # 3) Kết hợp CTC + AED
        loss_chunk = self.ctc_weight * loss_ctc \
                    + (1 - self.ctc_weight) * loss_att

        # 4) Tổng loss
        if loss_full is not None:
            loss = loss_full + loss_chunk
        else:
            loss = loss_chunk

        return loss, {'loss': loss, 'loss_ctc': loss_ctc, 'loss_att': loss_att}
```

**Giải thích & Lý do**

* **CTC loss**: dùng `nn.CTCLoss(blank=…, zero_infinity=True)` giống Wenet để bỏ qua NaN/Inf nếu có .
* **AED loss**: dùng LabelSmoothing (lsm\_weight) trên cross-entropy cho decoder output, cũng theo Wenet.
* **Reverse decoder** (`reverse_weight`) nếu bạn dùng bi-directional decoder (`bitransformer`).
* **ctc\_weight=0.3** và **1 – ctc\_weight** cho AED, đúng với cấu hình hybrid của Wenet .
* **Tách full-context vs chunk-context**: nếu bạn muốn kết hợp cả hai (dynamic training), gộp thêm `loss_full`. Còn nếu chỉ fine-tune chunk, bỏ phần full và để `loss = loss_chunk`.

Với hàm `forward` trả lại dict chứa cả các thành phần loss, bạn có thể log riêng `loss_ctc` và `loss_att` lên TensorBoard như Wenet làm trong `wenet/utils/train_utils.py`.

Như vậy bạn có đủ blueprint để implement loss cho ChunkFormer đúng theo “chuẩn” Wenet.
