
# chunkformer_vpb/finetune_utils.py

import torch
import argparse
from torchaudio.compliance.kaldi import fbank
from jiwer import wer
from ..decode import init, load_audio
from ..model.asr_model import ASRModel
from ..model.utils.ctc_utils import get_output_with_timestamps
from ..model.utils.mask import make_pad_mask
from torch.nn.utils.rnn import pad_sequence
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


    def build_y_batch(
        self,
        toks: torch.Tensor,       # [B, L_pad] no sos/eos
        tok_lens: torch.Tensor,   # [B]
        sos_id: int,
        eos_id: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Thêm sos/eos cho mỗi dòng, rồi pad về batch.
        Trả về:
          ys_pad:  [B, L_max+2]
          ys_lens: [B]
        """
        seqs = []
        lens = []
        B = toks.size(0)
        for i in range(B):
            raw = toks[i, : tok_lens[i]].tolist()
            seq = [sos_id] + raw + [eos_id]
            seqs.append(torch.tensor(seq, dtype=torch.long))
            lens.append(len(seq))
        ys_pad = pad_sequence(seqs, batch_first=True, padding_value=self.token2id.get("<pad>", 0))
        ys_lens = torch.tensor(lens, dtype=torch.long)
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


# import os, torch
# DEBUG_CHUNK = bool(int(os.getenv("DEBUG_CHUNK", "1")))

# @torch.no_grad()
# def _chunk_encoder_forward(xs: torch.Tensor,
#                            model,
#                            chunk_cfg,
#                            device) -> Tuple[torch.Tensor, torch.Tensor]:
#     """
#     Chạy encoder theo kiểu chunk + cache.
#     Trả về:
#         encoder_outs_full: [1, T_out, D]
#         encoder_mask     : [1, 1, T_out]
#     """
#     chunk_size   = chunk_cfg.chunk_size
#     left_context = chunk_cfg.left_context_size
#     right_context= chunk_cfg.right_context_size
#     subsample    = model.encoder.embed.subsampling_factor
#     conv_lorder  = model.encoder.cnn_module_kernel // 2
#     num_blocks   = model.encoder.num_blocks

#     # ─── Tính truncated + rel_right giống lúc decode ───
#     max_len   = int(chunk_cfg.total_batch_duration // 0.01) // 2
#     multiply_n = max_len // chunk_size // subsample
#     truncated   = chunk_size * multiply_n
#     rel_right   = (right_context + max(chunk_size, right_context)*(num_blocks-1)) * subsample

#     # ─── Init cache ───
#     offset    = torch.zeros(1, dtype=torch.int, device=device)
#     att_cache = torch.zeros((num_blocks, left_context,
#                              model.encoder.attention_heads,
#                              model.encoder._output_size*2 // model.encoder.attention_heads),
#                             device=device)
#     cnn_cache = torch.zeros((num_blocks, model.encoder._output_size, conv_lorder),
#                             device=device)

#     chunks = []
#     frame_idx = 0
#     while frame_idx < xs.size(1):
#         start = frame_idx
#         end   = min(start + truncated*subsample + 7, xs.size(1))
#         x     = xs[:, start:end + rel_right]     # lấy thêm right context
#         x_len = torch.tensor([x.size(1)], device=device)

#         if DEBUG_CHUNK:
#             print(f"[CHUNK] start={start:5d} end={end:5d} "
#                   f"x.shape={list(x.shape)} att_cache={list(att_cache.shape)}")

#         out, out_len, _, att_cache, cnn_cache, offset = model.encoder.forward_parallel_chunk(
#             xs=x,
#             xs_origin_lens=x_len,
#             chunk_size=chunk_size,
#             left_context_size=left_context,
#             right_context_size=right_context,
#             att_cache=att_cache,
#             cnn_cache=cnn_cache,
#             truncated_context_size=truncated,
#             offset=offset
#         )

#         if DEBUG_CHUNK:
#             print(f"        → out.shape={list(out.shape)} out_len={out_len.item()} "
#                   f"offset={offset.item()}")

#         chunks.append(out[:, :out_len])          # cắt đúng độ dài real
#         frame_idx += truncated * subsample
#         if frame_idx + rel_right >= xs.size(1):
#             break

#     # trong _chunk_encoder_forward, trước khi cat:
#     if DEBUG_CHUNK:
#         for i, ch in enumerate(chunks):
#             print(f"[DEBUG] chunk#{i} shape = {ch.shape}")

#     encoder_outs_full = torch.cat(chunks, dim=1)           # [1, T_all, D]
#     encoder_mask      = torch.ones(1, 1, encoder_outs_full.size(1), device=device)

#     if DEBUG_CHUNK:
#         print(f"[CHUNK] final encoder_outs_full {list(encoder_outs_full.shape)}")

#     return encoder_outs_full, encoder_mask


def full_encoder_forward(
    feats: torch.Tensor,        # [B, T_raw, D_feat]
    feat_lens: torch.Tensor,    # [B]
    model: ASRModel,
    cfg,
    device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Offline full-forward của encoder (không chunk).
    Trả về:
      enc_out:  [B, T_enc, D_model]
      enc_mask: [B, 1, T_enc]
    """
    # di chuyển device
    feats = feats.to(device)
    feat_lens = feat_lens.to(device)

    # gọi encoder trực tiếp (signature: encoder(xs, xs_lens))
    # nhiều encoder trả 3 giá trị: out, mask, something
    enc_out, enc_mask, *_ = model.encoder.forward(
        feats, 
        feat_lens, 
        limited_context_selection=[cfg.chunk.chunk_size,
                                   cfg.chunk.left_context_size,
                                   cfg.chunk.right_context_size]
    )

    # nếu enc_mask là [B, T], biến thành [B,1,T]
    if enc_mask.dim() == 2:
        enc_mask = enc_mask.unsqueeze(1)

    return enc_out, enc_mask


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
    ys_pad, ys_lens = tokenizer.text2labels(label_text)
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
import os 
DEBUG_LOSS = bool(int(os.getenv("DEBUG_LOSS", "1")))   # 1 → in log

def compute_loss_batch(
    model: ASRModel,
    feats: torch.Tensor,      # [B, T, D]
    feat_lens: torch.Tensor,  # [B]
    toks: torch.Tensor,       # [B, L_pad]   (no sos/eos)
    tok_lens: torch.Tensor,   # [B]
    cfg,
    device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

    B = feats.size(0)
    total_ctc, total_att = 0.0, 0.0
    sos, eos = model.sos, model.eos

    for i in range(B):
        x       = feats[i].unsqueeze(0)           # [1, T, D]
        y_raw   = toks[i, : tok_lens[i]].tolist()
        y_len   = tok_lens[i].unsqueeze(0).to(device)

        # ---- 1) encoder ----
        enc_out, enc_mask = _chunk_encoder_forward(x, model, cfg.chunk, device)
        enc_lens = enc_mask.squeeze(1).sum(1).unsqueeze(0).to(torch.long)  # [1]

        # print shape 
        # print (f"[DBG] sample {i}: enc_out {enc_out.shape}, enc_lens {enc_lens.tolist()}, "
        #        f"y_len {y_len.tolist()}")

        # vector‐hóa cả batch, offline:
        # enc_out, enc_mask = full_encoder_forward(
        #     feats, feat_lens, model, cfg.chunk, device
        # )
        # # enc_out: [B, T_enc, D], enc_mask: [B,1,T_enc]
        # enc_lens = enc_mask.squeeze(1).sum(1).to(torch.long)  # [B]

        
        
        if DEBUG_LOSS:
            print(f"[DBG] sample {i}: enc_out {enc_out.shape}, enc_lens {enc_lens.tolist()}, "
                  f"y_len {y_len.tolist()}")

        # ---- 2) CTC (no sos/eos) ----
        y_pad_ctc = torch.tensor(y_raw, dtype=torch.long, device=device).unsqueeze(0)  # [1, L]
        loss_ctc, _ = model.ctc(enc_out, enc_lens, y_pad_ctc, y_len)
        loss_ctc = loss_ctc.sum()

        # ---- 3) AED (add sos/eos) ----
        ys_pad = torch.tensor([sos] + y_raw + [eos], dtype=torch.long, device=device).unsqueeze(0)
        ys_len = torch.tensor([len(y_raw)+2], dtype=torch.long, device=device)
        loss_att, _ = model._calc_att_loss(enc_out, enc_mask, ys_pad, ys_len)
        loss_att = loss_att.sum()

        total_ctc += loss_ctc
        total_att += loss_att

    avg_ctc = total_ctc / B
    avg_att = total_att / B
    
    # 5) hybrid
    ctc_w = model.ctc_weight
    loss = ctc_w*avg_ctc + (1-ctc_w)*avg_att

    return loss, avg_ctc, avg_att


def compute_loss_batch(
    model: ASRModel,
    feats: torch.Tensor,      # [B, T, D_feat]
    feat_lens: torch.Tensor,  # [B]
    toks: torch.Tensor,       # [B, L_pad]   (no sos/eos)
    tok_lens: torch.Tensor,   # [B]
    cfg,
    device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Tính loss CTC + AED trên toàn batch, offline.
    """
    B = feats.size(0)
    sos, eos = model.sos, model.eos

    # 1) full forward encoder
    enc_out, enc_mask = full_encoder_forward(feats, feat_lens, model, cfg, device)
    # enc_out: [B, T_enc, D], enc_mask: [B,1,T_enc]
    enc_lens = enc_mask.squeeze(1).sum(1).to(torch.long)  # [B]

    # 2) CTC loss (no sos/eos)
    # expects: (hs, hs_lens, ys_pad, ys_lens)
    loss_ctc, _ = model.ctc(enc_out, enc_lens, toks, tok_lens)
    # CTCLoss returns mean over batch by default → khôi phục sum/avg theo batch
    # nếu reduction='mean' thì loss_ctc = sum_i (ctc_i) / B
    # ta tính avg_ctc = loss_ctc
    avg_ctc = loss_ctc

    # 3) AED loss (add sos/eos)
    tokenizer = cfg.tokenizer.tokenizer
    ys_pad, ys_lens = tokenizer.build_y_batch(toks, tok_lens, sos_id=sos, eos_id=eos)
    ys_pad, ys_lens = ys_pad.to(device), ys_lens.to(device)

    loss_att, _ = model._calc_att_loss(enc_out, enc_mask, ys_pad, ys_lens)
    # LabelSmoothingLoss returns sum over tokens if normalize_length=False,
    # or mean if normalize_length=True. Giả sử ta dùng sum, ta normalize:
    avg_att = loss_att / B

    # 4) hybrid
    w = cfg.model.ctc_weight
    loss = w * avg_ctc + (1 - w) * avg_att

    return loss, avg_ctc, avg_att
