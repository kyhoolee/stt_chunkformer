import torch
import torch.nn as nn
from wenet.transformer.ctc import CTC
from wenet.transformer.label_smoothing_loss import LabelSmoothingLoss
from wenet.utils.common import IGNORE_ID

class ChunkFormerFineTuneLoss(nn.Module):
    def __init__(self, model, config: dict):
        """
        model: instance của ASRModel đã chứa encoder, ctc, decoder, decoder._calc_att_loss, v.v.
        config: dict parsed từ config.yaml, chứa các mục ctc_conf, model_conf, tokenizer_conf.
        """
        super().__init__()
        self.model = model
        # lấy weight
        self.ctc_weight = config["model_conf"]["ctc_weight"]      # e.g. 0.3
        self.att_weight = 1.0 - self.ctc_weight

        # CTC : sử dụng chính CTC module đã gắn trong model
        # nhưng chúng ta khởi lại để chắc chắn zero_infinity=True
        blank_id = config["ctc_conf"]["ctc_blank_id"]
        self.ctc_loss_fn = CTC(blank_id=blank_id, zero_infinity=True)

        # AED : label smoothing
        lsm = config["model_conf"].get("lsm_weight", 0.0)
        vocab_size = config["tokenizer_conf"].get("vocab_size", model.vocab_size)
        self.att_loss_fn = LabelSmoothingLoss(
            size=vocab_size,
            padding_idx=IGNORE_ID,
            smoothing=lsm
        )

    def forward(self,
                feats: torch.Tensor,
                feat_lens: torch.Tensor,
                ys_pad: torch.Tensor,
                ys_lens: torch.Tensor):
        """
        feats: [B, T, D], feat_lens: [B]
        ys_pad: [B, L], padded token IDs including sos and eos
        ys_lens: [B], true lengths including sos & eos
        """
        # 1) chunked encoder forward (giống decode_aed_long_form but return mask)
        encoder_out, encoder_mask = self.model.encoder.forward_chunk_context(
            feats, feat_lens,
            chunk_size=self.model.encoder.chunk_size,
            left_context_size=self.model.encoder.left_context_size,
            right_context_size=self.model.encoder.right_context_size
        )
        # encoder_mask: [B,1,T_out], encoder_out: [B, T_out, D]
        encoder_out_lens = encoder_mask.squeeze(1).sum(1)  # [B]

        # 2) CTC loss via model.ctc (trong ASRModel, ctc returns (loss, metrics))
        loss_ctc, _ = self.model.ctc(
            encoder_out,
            encoder_out_lens,
            ys_pad,
            ys_lens
        )
        # loss_ctc: Tensor([B]) if reduction='none', or scalar if 'sum'

        # 3) AED loss via model._calc_att_loss
        #    returns (loss_att, accuracy)
        loss_att, _ = self.model._calc_att_loss(
            encoder_out,
            encoder_mask,
            ys_pad,
            ys_lens
        )
        # loss_att: Tensor([B]) or scalar

        # 4) Combine
        # ensure both are summed scalars
        loss_ctc = loss_ctc.sum() if loss_ctc.dim()>0 else loss_ctc
        loss_att = loss_att.sum() if loss_att.dim()>0 else loss_att

        loss = self.ctc_weight * loss_ctc + self.att_weight * loss_att

        return {
            "loss": loss,
            "loss_ctc": loss_ctc,
            "loss_att": loss_att
        }
