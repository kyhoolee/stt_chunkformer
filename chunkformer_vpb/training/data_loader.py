# modules/data_loader.py

import os
import json
import torchaudio
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from typing import List
from .finetune_config import FinetuneConfig
from chunkformer_vpb.training.finetune_utils import compute_fbank, MetadataEntry

class VivosDataset(Dataset):
    def __init__(self, cfg: FinetuneConfig, split: str):
        """
        cfg: FinetuneConfig đã load từ finetune_config.yaml
        split: one of "train", "valid" (or "dev"), "test"
        """
        self.cfg = cfg
        # path tới manifest_dir/train_meta.json, valid_meta.json, test_meta.json
        manifest_file = os.path.join(cfg.data.manifest_dir, f"{split}_meta.json")
        with open(manifest_file, 'r', encoding='utf-8') as f:
            entries = json.load(f)
        # Chuyển dict → MetadataEntry
        self.meta: List[MetadataEntry] = [MetadataEntry(**e) for e in entries]
        # GreedyTokenizer instance
        self.tokenizer = cfg.tokenizer.tokenizer

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        entry = self.meta[idx]
        # 1) Load waveform
        wav, sr = torchaudio.load(entry.audio_path)
        if sr != self.cfg.data.sample_rate:
            wav = torchaudio.transforms.Resample(sr, self.cfg.data.sample_rate)(wav)
        # 2) Extract features
        feats = compute_fbank(
            wav,
            num_mel_bins=self.cfg.data.num_mel_bins,
            frame_length=self.cfg.data.frame_length,
            frame_shift=self.cfg.data.frame_shift,
            dither=self.cfg.data.dither,
            energy_floor=self.cfg.data.energy_floor,
            sample_frequency=self.cfg.data.sample_rate
        )  # [T, D]
        # 3) Tokenize text → ids
        token_ids = self.tokenizer.tokenize(entry.text)
        return feats, feats.size(0), torch.LongTensor(token_ids), len(token_ids)

def collate_fn(batch):
    """Không thêm SOS/EOS – chỉ pad thô cho CTC"""
    feats, feat_lens, toks, tok_lens = zip(*batch)

    feats     = pad_sequence(feats, batch_first=True)          # [B, T_max, D]
    feat_lens = torch.LongTensor(feat_lens)                    # [B]
    toks      = pad_sequence(toks, batch_first=True, padding_value=0)  # [B, L_max]
    tok_lens  = torch.LongTensor(tok_lens)                     # [B]
    return feats, feat_lens, toks, tok_lens


def get_dataloaders(cfg: FinetuneConfig):
    """
    Trả về train_loader và valid_loader.
    cfg.training.batch_size được dùng làm batch_size.
    """
    bs = cfg.training.batch_size
    train_ds = VivosDataset(cfg, "train")
    valid_ds = VivosDataset(cfg, "valid")
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True,  collate_fn=collate_fn)
    valid_loader = DataLoader(valid_ds, batch_size=bs, shuffle=False, collate_fn=collate_fn)
    return train_loader, valid_loader
