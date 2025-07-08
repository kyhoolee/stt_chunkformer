from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import torch
import logging

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
        reverse_weight: float = 0.0,
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
