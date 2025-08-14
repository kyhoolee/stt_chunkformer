import torch
import torch.nn as nn
from wenet.transformer.ctc import CTC
from wenet.transformer.label_smoothing_loss import LabelSmoothingLoss
from wenet.utils.common import IGNORE_ID


# Copyright (c) 2019 Shigeki Karita
#               2020 Mobvoi Inc (Binbin Zhang)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Label smoothing module."""

import torch
from torch import nn


class LabelSmoothingLoss(nn.Module):
    """Label-smoothing loss.

    In a standard CE loss, the label's data distribution is:
    [0,1,2] ->
    [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ]

    In the smoothing version CE Loss,some probabilities
    are taken from the true label prob (1.0) and are divided
    among other labels.

    e.g.
    smoothing=0.1
    [0,1,2] ->
    [
        [0.9, 0.05, 0.05],
        [0.05, 0.9, 0.05],
        [0.05, 0.05, 0.9],
    ]

    Args:
        size (int): the number of class
        padding_idx (int): padding class id which will be ignored for loss
        smoothing (float): smoothing rate (0.0 means the conventional CE)
        normalize_length (bool):
            normalize loss by sequence length if True
            normalize loss by batch size if False
    """

    def __init__(self,
                 size: int,
                 padding_idx: int,
                 smoothing: float,
                 normalize_length: bool = False):
        """Construct an LabelSmoothingLoss object."""
        super(LabelSmoothingLoss, self).__init__()
        self.criterion = nn.KLDivLoss(reduction="none")
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.normalize_length = normalize_length

    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute loss between x and target.

        The model outputs and data labels tensors are flatten to
        (batch*seqlen, class) shape and a mask is applied to the
        padding part which should not be calculated for loss.

        Args:
            x (torch.Tensor): prediction (batch, seqlen, class)
            target (torch.Tensor):
                target signal masked with self.padding_id (batch, seqlen)
        Returns:
            loss (torch.Tensor) : The KL loss, scalar float value
        """
        assert x.size(2) == self.size
        batch_size = x.size(0)
        x = x.view(-1, self.size)
        # target = target.view(-1)
        target = target.reshape(-1)

        # use zeros_like instead of torch.no_grad() for true_dist,
        # since no_grad() can not be exported by JIT
        true_dist = torch.zeros_like(x)
        true_dist.fill_(self.smoothing / (self.size - 1))
        ignore = target == self.padding_idx  # (B,)
        total = len(target) - ignore.sum().item()
        target = target.masked_fill(ignore, 0)  # avoid -1 index
        true_dist.scatter_(1, target.unsqueeze(1), self.confidence)
        kl = self.criterion(torch.log_softmax(x, dim=1), true_dist)
        denom = total if self.normalize_length else batch_size
        return kl.masked_fill(ignore.unsqueeze(1), 0).sum() / denom


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
