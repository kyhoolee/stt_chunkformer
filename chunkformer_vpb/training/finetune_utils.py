
# chunkformer_vpb/finetune_utils.py

import torch
import argparse
from torchaudio.compliance.kaldi import fbank
from jiwer import wer
from ..decode import init, load_audio
from ..model.asr_model import ASRModel
from ..model.utils.ctc_utils import get_output_with_timestamps
from ..model.utils.mask import make_pad_mask
from typing import Dict, List, Tuple
from typing import Tuple
from typing import TYPE_CHECKING



# ==================== ARG PARSE ====================
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_checkpoint', type=str, required=True,
                        help='Path to model checkpoint folder (e.g. chunkformer-large-vie)')
    parser.add_argument('--audio_path', type=str, required=True,
                        help='Path to .wav audio file to transcribe')
    parser.add_argument('--label_text', type=str, default=None,
                        help='(Optional) Ground truth text for WER comparison')
    parser.add_argument('--chunk_size', type=int, default=64)
    parser.add_argument('--left_context_size', type=int, default=128)
    parser.add_argument('--right_context_size', type=int, default=128)
    parser.add_argument('--total_batch_duration', type=int, default=1800)  # in ms
    args = parser.parse_args()
    return args

def get_default_args():
    return argparse.Namespace(
        model_checkpoint="chunkformer-large-vie",
        audio_path="samples/test.wav",
        label_text=None,
        chunk_size=64,
        left_context_size=128,
        right_context_size=128,
        total_batch_duration=1800
    )

# ==================== MODEL ====================
def load_model_only(model_checkpoint="../chunkformer-large-vie", device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, char_dict = init(model_checkpoint, device)
    model.eval()
    return model, char_dict

def dump_module_structure(module: torch.nn.Module, output_path: str, title: str = ""):
    with open(output_path, "w", encoding="utf-8") as f:
        if title:
            f.write(f"==== {title} ====\n\n")
        f.write(str(module))


# ==================== PREPROCESS ====================
def prepare_input_file(audio_path, device):
    waveform = load_audio(audio_path)
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    if waveform.abs().max() <= 1.0:
        waveform = waveform * 32768.0
    waveform = waveform.clamp(-32768, 32767).to(device)

    feats = fbank(
        waveform,
        num_mel_bins=80,
        frame_length=25,
        frame_shift=10,
        dither=0.0,
        energy_floor=0.0,
        sample_frequency=16000
    ).unsqueeze(0).to(device)
    return feats



# ==================== TOKENIZER UTILS ====================

# class GreedyTokenizer:
#     def __init__(self, vocab_path: str):
#         """
#         vocab_path: đường dẫn tới file vocab.txt (mỗi dòng 1 token)
#         """
#         self.vocab = []
#         with open(vocab_path, encoding="utf-8") as f:
#             for line in f:
#                 token = line.strip().split()[0]
#                 self.vocab.append(token)
#         self.token2id = {tok: idx for idx, tok in enumerate(self.vocab)}
#         # sort decreasing độ dài để greedy match
#         self.vocab_sorted = sorted(self.vocab, key=len, reverse=True)

#     def tokenize(self, text: str) -> List[int]:
#         """
#         Greedy longest-match tokenization.
#         - Thêm '▁' đầu câu, thay space bằng '▁'.
#         - Mỗi bước match token dài nhất.
#         """
#         s = "▁" + text.strip().replace(" ", "▁")
#         idx = 0
#         L = len(s)
#         ids = []
#         while idx < L:
#             for tok in self.vocab_sorted:
#                 if s.startswith(tok, idx):
#                     ids.append(self.token2id[tok])
#                     idx += len(tok)
#                     break
#             else:
#                 # fallback: <unk> nếu có, hoặc skip 1 char
#                 unk = self.token2id.get("<unk>")
#                 if unk is not None:
#                     ids.append(unk)
#                 idx += 1
#         return ids

#     def text2labels(self,
#                     text: str,
#                     sos_id: int,
#                     eos_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
#         """
#         Convert raw text → ys_pad, ys_lens tensors for loss calculation.
#         """
#         ids = self.tokenize(text)
#         seq = [sos_id] + ids + [eos_id]
#         ys_pad = torch.tensor(seq, dtype=torch.long).unsqueeze(0)  # [1, L]
#         ys_lens = torch.tensor([len(seq)], dtype=torch.long)      # [1]
#         return ys_pad, ys_lens
    
class GreedyTokenizer:
    def __init__(self, vocab_path: str):
        """
        vocab_path: đường dẫn tới file vocab.txt (mỗi dòng 1 token)
        """
        # ---------- load vocab ----------
        self.vocab = []
        with open(vocab_path, encoding="utf-8") as f:
            for line in f:
                token = line.strip().split()[0]
                self.vocab.append(token)

        # ---------- basic map ----------
        self.token2id: Dict[str, int] = {tok: idx for idx, tok in enumerate(self.vocab)}
        self.id2token: List[str] = self.vocab
        # sort decreasing độ dài để greedy match khi encode
        self.vocab_sorted = sorted(self.vocab, key=len, reverse=True)

        # ---------- special IDs ----------
        # giả định chuẩn Wenet: 0 = <blank> = pad, cuối vocab = <sos/eos>
        self.blank_id = 0
        self.pad_id   = 0                    # pad dùng cùng giá trị với blank
        self.sos_id   = len(self.vocab) - 1  # eos = sos
        self.eos_id   = self.sos_id

    # ---------- ENCODE ----------
    def tokenize(self, text: str) -> List[int]:
        """
        Greedy longest-match tokenization.
        - Thêm '▁' đầu câu, thay space bằng '▁'.
        """
        s = "▁" + text.strip().replace(" ", "▁")
        idx, L, ids = 0, len(s), []
        while idx < L:
            for tok in self.vocab_sorted:
                if s.startswith(tok, idx):
                    ids.append(self.token2id[tok])
                    idx += len(tok)
                    break
            else:
                # fallback: <unk> nếu có, hoặc skip 1 char
                ids.append(self.token2id.get("<unk>", self.blank_id))
                idx += 1
        return ids

    # ---------- DECODE ----------
    def decode_ids(self, ids: List[int]) -> str:
        """
        Convert list[int] → raw string, bỏ blank/pad/sos/eos.
        """
        tokens = [
            self.id2token[i]
            for i in ids
            if i not in {self.blank_id, self.pad_id, self.sos_id, self.eos_id}
        ]
        text = "".join(tokens).replace("▁", " ").strip()
        return text

    # ---------- helper cho AED ----------
    def text2labels(self, text: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return ys_pad & ys_lens (đã thêm sos/eos) để dùng cho AED.
        """
        ids = self.tokenize(text)
        seq = [self.sos_id] + ids + [self.eos_id]
        ys_pad  = torch.tensor(seq, dtype=torch.long).unsqueeze(0)   # [1, L]
        ys_lens = torch.tensor([len(seq)], dtype=torch.long)
        return ys_pad, ys_lens




# ==================== Forward loss UTILS ====================

@torch.no_grad()
def _chunk_encoder_forward(xs: torch.Tensor,
                           model,
                           chunk_cfg,
                           device) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Internal helper: chunk the input features xs through the encoder (CTC mode)
    Returns:
      encoder_outs_full: [1, T_out, D]
      encoder_mask:     [1, 1, T_out]
    """
    chunk_size = chunk_cfg.chunk_size
    left_context = chunk_cfg.left_context_size
    right_context = chunk_cfg.right_context_size
    subsample = model.encoder.embed.subsampling_factor
    conv_lorder = model.encoder.cnn_module_kernel // 2
    num_blocks = model.encoder.num_blocks

    # calculate truncated_context_size same as decode
    max_len = int(chunk_cfg.total_batch_duration // 0.01) // 2
    multiply_n = max_len // chunk_size // subsample
    truncated = chunk_size * multiply_n
    rel_right = (right_context + max(chunk_size, right_context)*(num_blocks-1))*subsample

    # init caches
    offset = torch.zeros(1, dtype=torch.int, device=device)
    att_cache = torch.zeros((num_blocks, left_context,
                             model.encoder.attention_heads,
                             model.encoder._output_size*2//model.encoder.attention_heads),
                            device=device)
    cnn_cache = torch.zeros((num_blocks,
                             model.encoder._output_size,
                             conv_lorder), device=device)

    chunks = []
    for idx in range(0, xs.shape[1], truncated*subsample):
        start = truncated*subsample*idx
        end = min(start + truncated*subsample + 7, xs.shape[1])
        x = xs[:, start:end+rel_right]
        x_len = torch.tensor([x.shape[1]], device=device)
        out, out_len, _, att_cache, cnn_cache, offset = model.encoder.forward_parallel_chunk(
            xs=x,
            xs_origin_lens=x_len,
            chunk_size=chunk_size,
            left_context_size=left_context,
            right_context_size=right_context,
            att_cache=att_cache,
            cnn_cache=cnn_cache,
            truncated_context_size=truncated,
            offset=offset
        )
        chunks.append(out[:, :out_len])
        if start + rel_right >= xs.shape[1]:
            break

    encoder_outs_full = torch.cat(chunks, dim=1)
    encoder_mask = torch.ones(1, 1, encoder_outs_full.size(1), device=device)
    return encoder_outs_full, encoder_mask


def compute_chunkformer_loss(model: ASRModel,
                             tokenizer: GreedyTokenizer,
                             xs: torch.Tensor,
                             args,
                             label_text: str,
                             device) -> dict:
    """
    xs: [1, T_raw, 80] input features
    args: arg namespace with chunk_size, left_context_size, right_context_size, total_batch_duration
    label_text: ground-truth string
    Returns: dict {loss, loss_ctc, loss_att}
    """
    # 1) forward chunk-encoder
    encoder_outs_full, encoder_mask = _chunk_encoder_forward(xs, model, args, device)
    # encoder_lens = encoder_mask.squeeze(1).sum(2)  # [1]
    # encoder_mask: [1, 1, T] → squeeze → [1, T], sum over time dim → [1]
    encoder_lens = encoder_mask.squeeze(1).sum(1).to(torch.long)

    # 2) text -> labels
    sos_id = model.sos
    eos_id = model.eos
    ys_pad, ys_lens = tokenizer.text2labels(label_text, sos_id, eos_id)
    ys_pad = ys_pad.to(device)
    ys_lens = ys_lens.to(device).to(torch.long)
    # @NOTE: 
    # - Phần chèn sos và eos này chỉ dùng cho AED loss 
    # - Không dùng cho CTC loss -> add thêm vào tăng giá trị không cần thiết của CTC loss
    # - Dùng thì loss vẫn giảm dần -> nhưng không đúng với CTC loss

    # 3) CTC loss
    loss_ctc, _ = model.ctc(
        encoder_outs_full,
        encoder_lens,
        ys_pad,
        ys_lens
    )
    # 4) AED loss
    loss_att, _ = model._calc_att_loss(
        encoder_outs_full,
        encoder_mask,
        ys_pad,
        ys_lens
    )

    # sum if needed
    loss_ctc = loss_ctc.sum() if loss_ctc.dim()>0 else loss_ctc
    loss_att = loss_att.sum() if loss_att.dim()>0 else loss_att

    # 5) hybrid
    ctc_w = model.ctc_weight
    loss = ctc_w*loss_ctc + (1-ctc_w)*loss_att

    return {"loss": loss,
            "loss_ctc": loss_ctc,
            "loss_att": loss_att}



# ==================== LOSS COMPUTE UTILS ====================

def compute_loss_batch(
    model: ASRModel,
    feats: torch.Tensor,      # [B, T, D]
    feat_lens: torch.Tensor,  # [B]
    toks: torch.Tensor,       # [B, L_pad]   (không sos/eos)
    tok_lens: torch.Tensor,   # [B]
    cfg,
    device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

    B = feats.size(0)
    total_ctc, total_att = 0.0, 0.0
    sos = model.sos
    eos = model.eos
    ignore = model.ignore_id

    for i in range(B):
        x       = feats[i].unsqueeze(0)      # [1, T, D]
        y_raw   = toks[i, : tok_lens[i]].tolist()              # list[int]
        x_len   = feat_lens[i].unsqueeze(0)
        y_len   = tok_lens[i].unsqueeze(0)

        # ---- 1) forward encoder (chunk) ----
        enc_out, enc_mask = _chunk_encoder_forward(x, model, cfg.chunk, device)
        enc_lens = enc_mask.squeeze(1).sum(1).to(torch.long)

        # ---- 2) CTC loss (NO sos/eos) ----
        y_pad_ctc = torch.LongTensor(y_raw).unsqueeze(0).to(device)   # [1, L]
        loss_ctc, _ = model.ctc(enc_out, enc_lens, y_pad_ctc, y_len)
        loss_ctc = loss_ctc.sum()

        # ---- 3) AED loss (ADD sos/eos) ----
        ys_pad = torch.LongTensor([sos] + y_raw + [eos]).unsqueeze(0).to(device)  # [1, L+2]
        ys_len = torch.LongTensor([len(y_raw)+2]).to(device)
        loss_att, _ = model._calc_att_loss(enc_out, enc_mask, ys_pad, ys_len)
        loss_att = loss_att.sum()

        total_ctc += loss_ctc
        total_att += loss_att

    avg_ctc = total_ctc / B
    avg_att = total_att / B
    w = cfg.model.ctc_weight
    loss = w * avg_ctc + (1 - w) * avg_att
    return loss, avg_ctc, avg_att
