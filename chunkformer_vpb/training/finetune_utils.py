
# chunkformer_vpb/finetune_utils.py

import torch
import argparse
from torchaudio.compliance.kaldi import fbank
from jiwer import wer

from .data_loader import normalize_vi
from ..decode import init, load_audio
from ..model.asr_model import ASRModel
from ..model.utils.ctc_utils import get_output_with_timestamps
from ..model.utils.mask import make_pad_mask
from torch.nn.utils.rnn import pad_sequence
from typing import Dict, List, Tuple
from typing import Tuple
from typing import TYPE_CHECKING
from .tokenizer import GreedyTokenizer



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
    Tính loss cho 1 sample:
    - xs: [1, T_raw, 80] input features
    - args: arg namespace với các trường chunk_size, left_context_size, ...
    - label_text: ground-truth dạng string
    Returns: dict với loss, loss_ctc, loss_att
    """
    # 1) Forward encoder
    encoder_outs_full, encoder_mask = _chunk_encoder_forward(xs, model, args, device)
    encoder_lens = encoder_mask.squeeze(1).sum(1).to(torch.long)  # [1]

    # 2) Tokenize raw text → list of IDs (không có sos/eos)
    norm_text = normalize_vi(label_text)
    token_ids = tokenizer.tokenize(norm_text)  # List[int]

    # CTC labels (no sos/eos)
    toks = torch.tensor(token_ids, dtype=torch.long, device=device).unsqueeze(0)  # [1, L]
    tok_lens = torch.tensor([len(token_ids)], dtype=torch.long, device=device)    # [1]

    # AED labels (có sos/eos)
    seq = [model.sos] + token_ids + [model.eos]
    ys_pad = torch.tensor(seq, dtype=torch.long, device=device).unsqueeze(0)      # [1, L+2]
    ys_lens = torch.tensor([len(seq)], dtype=torch.long, device=device)           # [1]

    # 3) CTC loss
    loss_ctc, _ = model.ctc(
        encoder_outs_full,
        encoder_lens,
        toks,
        tok_lens
    )

    # 4) AED loss
    loss_att, _ = model._calc_att_loss(
        encoder_outs_full,
        encoder_mask,
        ys_pad,
        ys_lens
    )

    # 5) Sum & combine
    loss_ctc = loss_ctc.sum() if loss_ctc.dim() > 0 else loss_ctc
    loss_att = loss_att.sum() if loss_att.dim() > 0 else loss_att

    ctc_w = model.ctc_weight
    loss = ctc_w * loss_ctc + (1 - ctc_w) * loss_att

    return {
        "loss": loss,
        "loss_ctc": loss_ctc,
        "loss_att": loss_att
    }

# ==================== LOSS COMPUTE UTILS ====================

import os
import torch
import torch.nn.functional as F
from typing import Tuple

DEBUG_LOSS = bool(int(os.getenv("DEBUG_LOSS", "0")))   # 1 → in log

import torch
import torch.nn.functional as F
from typing import Tuple
from torch.nn.utils.rnn import pad_sequence


def make_pad_mask(lengths: torch.Tensor, max_len: int = None) -> torch.Tensor:
    """
    Tạo mask từ vector length. Return shape: [B, 1, T]  #NOTE_PAD
    """
    B = lengths.size(0)
    max_len = max_len or lengths.max().item()
    ids = torch.arange(0, max_len, device=lengths.device).unsqueeze(0).expand(B, -1)
    mask = (ids < lengths.unsqueeze(1)).unsqueeze(1)  # [B, 1, T]
    return mask

def merge_chunks_by_sample(
    enc_outs: torch.Tensor,     # [total_chunks, chunk_len, D]
    batch_ids: torch.Tensor,    # [total_chunks], int64, giá trị từ 0 → B-1
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Gộp các chunk lại theo từng sample trong batch.
    Trả về: padded_outs: [B, T_max, D], lens: [B]
    """
    B = batch_ids.max().item() + 1
    D = enc_outs.shape[2]
    grouped_outs, lens = [], []

    for i in range(B):
        idxs = (batch_ids == i).nonzero().squeeze(1)
        out_i = enc_outs[idxs]      # [n_chunks_i, chunk_len, D]
        out_i = out_i.reshape(-1, D)
        grouped_outs.append(out_i)
        lens.append(out_i.shape[0])

    padded_outs = pad_sequence(grouped_outs, batch_first=True)  # [B, T_max, D]
    return padded_outs, torch.tensor(lens, dtype=torch.long, device=enc_outs.device)

def encode_offline_batch_debug(
    feats: torch.Tensor,       # [B, T_raw, D]
    feat_lens: torch.Tensor,   # [B]
    model,
    chunk_cfg,
    device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Như encode_offline_batch nhưng chèn debug prints:
      - In shape đầu vào
      - In shape sau mỗi sample
      - In pad time và mask pad (chỉ khi B > 1)
      - In shape cuối của enc_outs và enc_masks
    """
    B, T_raw, D = feats.shape
    print(f"[ENC_OFFLINE] start: B={B}, T_raw_max={T_raw}, D={D}")

    all_chunks, all_masks, batch_ids = [], [], []

    for i in range(B):
        x_i = feats[i : i+1].to(device)       # [1, T_raw, D]
        l_i = feat_lens[i].item()
        print(f"  [sample {i}] raw_feat.shape = {x_i.shape}, raw_len = {l_i}")

        out_i, m_i = _chunk_encoder_forward(x_i, model, chunk_cfg, device)
        # out_i: [N_chunk_i, chunk_len, D], m_i: [1, 1, T_chunk]
        print(f"    → after chunk-fwd: out_i.shape = {out_i.shape}, mask_i.shape = {m_i.shape}")

        all_chunks.append(out_i)                        # [N_chunk_i, chunk_len, D]
        all_masks.append(m_i)                           # m_i ở dạng [1, 1, T]
        batch_ids += [i] * out_i.shape[0]               # mỗi chunk đánh dấu thuộc sample nào

    enc_outs_cat = torch.cat(all_chunks, dim=0)         # [N_total_chunks, chunk_len, D]
    batch_ids = torch.tensor(batch_ids, device=device)  # [N_total_chunks]



    # Nếu batch > 1 → gộp chunk từng sample, pad đều
    enc_outs, enc_lens = merge_chunks_by_sample(enc_outs_cat, batch_ids)  # [B, T_max, D], [B]
    T_max = enc_outs.shape[1]
    print(f"[ENC_OFFLINE] pad to T_max = {T_max}")

    # Tạo lại mask theo lens
    # enc_masks = torch.zeros(B, 1, T_max, device=device)
    enc_masks = make_pad_mask(enc_lens, T_max)          #NOTE_PAD
    # for i in range(B):
    #     enc_masks[i, 0, :enc_lens[i]] = 1

    print(f"[ENC_OFFLINE] final enc_outs.shape = {enc_outs.shape}, enc_masks.shape = {enc_masks.shape}")
    return enc_outs, enc_masks


def compute_loss_batch_v1(
    model: ASRModel,
    feats: torch.Tensor,      # [B, T_raw, D]
    feat_lens: torch.Tensor,  # [B]
    toks: torch.Tensor,       # [B, L_pad]   (no sos/eos)
    tok_lens: torch.Tensor,   # [B]
    cfg,
    device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

    # 1) Offline‐encode cả batch
    enc_outs, enc_masks = encode_offline_batch_debug(feats, feat_lens, model, cfg.chunk, device)
    # enc_outs: [B, T_enc, D], enc_masks: [B,1,T_enc]
    enc_lens = enc_masks.squeeze(1).sum(1).to(torch.long)  # [B]

    if DEBUG_LOSS:
        print(f"[DBG] enc_outs {enc_outs.shape}, enc_lens {enc_lens.tolist()}")

    # Token debug
    print(f"[DBG] toks.shape={toks.shape}, tok_lens={tok_lens.tolist()}")
    for i in range(toks.size(0)):
        raw = toks[i, : tok_lens[i]].tolist()
        print(f"[DBG] sample {i}: toks[:{tok_lens[i]}] = {raw}")


    # 2) CTC loss (không sos/eos)
    #   model.ctc expects: (logp, input_lengths, targets, target_lengths)
    loss_ctc, _ = model.ctc(
        enc_outs, enc_lens,
        toks.to(device), tok_lens.to(device)
    )
    # sum across batch
    loss_ctc = loss_ctc.sum()

    print(f"[DBG] loss_ctc = {loss_ctc.item():.4f}")

    # 3) AED loss (cần sos/eos)
    # build ys_pad, ys_lens cho batch
    sos_id, eos_id = model.sos, model.eos
    # create [B, L+2] and lengths [B]
    ys_pad = []
    ys_lens = []
    for i in range(feats.size(0)):
        raw = toks[i, : tok_lens[i]].tolist()
        seq = [sos_id] + raw + [eos_id]
        ys_pad.append(torch.tensor(seq, dtype=torch.long, device=device))
        ys_lens.append(len(seq))
    ys_pad = torch.nn.utils.rnn.pad_sequence(ys_pad, batch_first=True, padding_value=model.ignore_id)  # [B, L_max+2]
    ys_lens = torch.tensor(ys_lens, dtype=torch.long, device=device)                                 # [B]

    print(f"[DBG] ys_pad.shape={ys_pad.shape}, ys_lens={ys_lens.tolist()}")


    loss_att, _ = model._calc_att_loss(enc_outs, enc_masks, ys_pad, ys_lens)
    loss_att = loss_att.sum()
    print(f"[DBG] loss_att = {loss_att.item():.4f}")

    # 4) Hybrid
    ctc_w = cfg.model.ctc_weight
    loss = ctc_w * (loss_ctc / feats.size(0)) + (1 - ctc_w) * (loss_att / feats.size(0))


    print(f"[DBG] final loss = {loss.item():.4f}")
    return loss, (loss_ctc / feats.size(0)), (loss_att / feats.size(0))



# ======================================================================================================================

def compute_loss_batch_v2(
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
