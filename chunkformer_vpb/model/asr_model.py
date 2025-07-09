from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import torch
import logging
import torch.nn.functional as F


from wenet.transformer.label_smoothing_loss import LabelSmoothingLoss
from torch.nn.utils.rnn import pad_sequence
from .ctc import CTC
from .utils.common import (IGNORE_ID, add_sos_eos, log_add,
                                remove_duplicates_and_blank, th_accuracy,
                                reverse_pad_list)
# from .mwer import MWER
from .utils.mask import (make_pad_mask, mask_finished_preds,
                              mask_finished_scores, subsequent_mask)


class ASRModel(torch.nn.Module):
    """CTC-attention hybrid Encoder-Decoder model"""
    def __init__(
        self,
        vocab_size: int,
        encoder,
        ctc: CTC,
        ctc_weight: float = 0.5,
        ignore_id: int = IGNORE_ID,
        reverse_weight: float = 0.0, lsm_weight: float=0.1,
        decoder=None
    ):
        assert 0.0 <= ctc_weight <= 1.0, ctc_weight

        super().__init__()
        # note that eos is the same as sos (equivalent ID)
        self.sos = vocab_size - 1
        self.eos = vocab_size - 1
        self.vocab_size = vocab_size
        self.ignore_id = ignore_id
        self.ctc_weight = ctc_weight
        self.reverse_weight = reverse_weight

        self.encoder = encoder
        self.encoder.ctc = ctc
        self.ctc = ctc        
        self.decoder = decoder

        # add label‐smoothing loss for AED
        self.criterion_att = LabelSmoothingLoss(
            size=vocab_size,
            padding_idx=ignore_id,
            smoothing=lsm_weight,
            normalize_length=False
        )

    def decode_aed(self, encoder_out: torch.Tensor, encoder_mask: torch.Tensor, maxlen: int = 100):
        """
        Greedy AED decoding:  
          - encoder_out: [B, T_in, D]  
          - encoder_mask: [B, 1, T_in]  
        Trả về:  
          - token IDs [B, T_out], score sequence [B, T_out, V]
        """
        batch_size = encoder_out.size(0)
        # Khởi tạo ys với sos
        ys = encoder_out.new_full((batch_size, 1), fill_value=self.sos, dtype=torch.long)
        scores = []

        for _ in range(maxlen):
            # Tạo tensor độ dài của ys
            ys_lens = torch.tensor([ys.size(1)] * batch_size, dtype=torch.long, device=ys.device)
            # Gọi decoder với đúng thứ tự args
            l_x, r_x, olens = self.decoder(
                encoder_out,    # memory
                encoder_mask,   # memory_mask
                ys,             # ys_in_pad
                ys_lens,        # ys_in_lens
                None,           # r_ys_in_pad (không dùng reverse here)
                0.0,            # reverse_weight
            )
            # l_x: [B, T_cur, V]
            # Lấy logits của token cuối cùng
            prob = torch.nn.functional.log_softmax(l_x[:, -1], dim=-1)
            next_token = prob.argmax(dim=-1, keepdim=True)  # [B,1]
            ys = torch.cat([ys, next_token], dim=1)         # [B, T_cur+1]
            scores.append(prob)

            # Nếu tất cả batch đều sinh eos thì dừng sớm
            if (next_token == self.eos).all():
                break

        # Trả về sequence bỏ sos + scores
        return ys[:, 1:], torch.stack(scores, dim=1)


    def _calc_att_loss(
        self,
        encoder_out: torch.Tensor,    # [B, T_enc, D]
        encoder_mask: torch.Tensor,   # [B, 1, T_enc]
        ys_pad: torch.Tensor,         # [B, L] chứa <sos>…<eos>
        ys_lens: torch.Tensor         # [B]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute AED loss w/ label smoothing + token‐level accuracy.
        """
        # 1) prepare ys_in, ys_out
        ys_in  = ys_pad[:, :-1]    # drop <eos>
        ys_out = ys_pad[:, 1:]     # drop <sos>
        ys_in_lens = (ys_in != self.ignore_id).sum(1)

        # 2) decoder forward
        decoder_out, _, _ = self.decoder(
            encoder_out, encoder_mask, ys_in, ys_in_lens, None, self.reverse_weight
        )
        # 3) log-probs
        logp = F.log_softmax(decoder_out, dim=-1)  # [B, T, V]

        # 4) AED loss
        loss_att = self.criterion_att(logp, ys_out)

        # 5) accuracy
        # B, T, V = logp.size()
        # pred_flat   = logp.reshape(-1, V)
        # target_flat = ys_out.reshape(-1)
        # acc_att     = th_accuracy(pred_flat, target_flat, ignore_label=self.ignore_id)

        return loss_att, None