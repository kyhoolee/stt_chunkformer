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
        """Greedy decoding using AED"""
        batch_size = encoder_out.size(0)
        ys = encoder_out.new_full((batch_size, 1), fill_value=self.sos, dtype=torch.long)

        scores = []
        for i in range(maxlen):
            decoder_out, _ = self.decoder(ys, encoder_out, encoder_mask)
            prob = torch.nn.functional.log_softmax(decoder_out[:, -1], dim=-1)
            next_token = prob.argmax(dim=-1, keepdim=True)
            ys = torch.cat([ys, next_token], dim=1)
            scores.append(prob)
            if (next_token == self.eos).all():
                break

        return ys[:, 1:], torch.stack(scores, dim=1)  # remove <sos>
