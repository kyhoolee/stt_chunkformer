# modules/data_loader.py

import os
import json
import torchaudio
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from typing import List
from dataclasses import fields


from ..model.utils.common import IGNORE_ID
from .finetune_config import FinetuneConfig
from ..data.data import compute_fbank, MetadataEntry
from .tokenizer import normalize_vi
from .data_augment import AudioAugmenter

# BẬT/TẮT DEBUG IN INFO TRONG COLLATE
DEBUG_COLLATE = True

class VivosDataset(Dataset):
    def __init__(self, cfg: FinetuneConfig, split: str):
        self.cfg = cfg
        print("==================================")
        print(self.cfg.data.audio_dir)
        print("==================================")
        meta_file = f"{split}_meta.json"

        print(f"{cfg.data}")
        if split == "train" and cfg.data.train_meta_file:
            meta_file = cfg.data.train_meta_file

        manifest_file = os.path.join(cfg.data.manifest_dir, meta_file)
        with open(manifest_file, 'r', encoding='utf-8') as f:
            entries = json.load(f)
        # dict → MetadataEntry
        # self.meta: List[MetadataEntry] = [MetadataEntry(**e) for e in entries]
        # Lấy danh sách field hợp lệ từ dataclass
        valid_keys = {f.name for f in fields(MetadataEntry)}

        # Lọc từng dict trước khi tạo MetadataEntry
        self.meta = [
            MetadataEntry(**{k: v for k, v in e.items() if k in valid_keys})
            for e in entries
        ]
        self.tokenizer = cfg.tokenizer.tokenizer

        self.augmenter = AudioAugmenter(cfg.data.sample_rate)

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        entry = self.meta[idx]
        # 1) Load waveform
        audio_dir = self.cfg.data.audio_dir 
        if not audio_dir:
            audio_dir = self.cfg.data.manifest_dir
        abs_path = audio_dir + os.sep + entry.audio_path
        wav, sr = torchaudio.load(abs_path)

        
        # print(f"✅ [loader] Audio path       : {entry.audio_path}")
        # print(f"📏 [loader] Sample rate      : {sr}")
        # print(f"   [loader] wav.shape: {wav.shape}, sample_rate: {sr}")
        if sr != self.cfg.data.sample_rate:
            wav = torchaudio.transforms.Resample(sr, self.cfg.data.sample_rate)(wav)

        if wav.dim() == 1:
            wav = wav.unsqueeze(0)
        if wav.abs().max() <= 1.0:
            wav = wav * 32768.0
        wav = wav.clamp(-32768, 32767)

        # print(f"✅ [loader] Waveform shape    : {wav.shape}")
        # print(f"📊 [loader] Min: {wav.min().item():.8f}, Max: {wav.max().item():.8f}, Mean: {wav.mean().item():.8f}")

        # @NOTE: bỏ việc augment audio trong collate_fn -> theo hướng pre-compute trước augmented audio 
        # 🧪 Apply augmentation nếu có "augment_type"
        # augment_type = entry.augment_type
        # if augment_type:
        #     wav = self.augmenter.apply(wav, augment_type)

        # 2) FBANK
        feats = torchaudio.compliance.kaldi.fbank(
            wav,
            num_mel_bins = self.cfg.data.num_mel_bins,
            frame_length = self.cfg.data.frame_length,
            frame_shift  = self.cfg.data.frame_shift,
            dither       = self.cfg.data.dither,
            energy_floor = self.cfg.data.energy_floor,
            sample_frequency = self.cfg.data.sample_rate
        )  # [T, D]
        # 3) Tokenize text
        norm_text = normalize_vi(entry.text)
        token_ids = self.tokenizer.tokenize(norm_text)
        toks = torch.LongTensor(token_ids)
        return feats, feats.size(0), toks, len(token_ids), (entry, wav)

def collate_fn(batch):
    """
    batch: list of tuples (feats, feat_len, toks, tok_len, entry)
    """
    feats, feat_lens, toks, tok_lens, entries = zip(*batch)

    # if DEBUG_COLLATE:
    #     for i, e in enumerate(entries):
    #         print(f"[collate] sample {i}: utt_id={e.utt_id}, audio={e.audio_path}")

    feats     = pad_sequence(feats, batch_first=True, padding_value=0)                   # [B, T_max, D]
    feat_lens = torch.LongTensor(feat_lens)                             # [B]
    toks      = pad_sequence(toks, batch_first=True, padding_value=0)    # [B, L_max]
    tok_lens  = torch.LongTensor(tok_lens)                               # [B]

    return feats, feat_lens, toks, tok_lens

def collate_fn_entry(batch):
    """
    batch: list of tuples (feats, feat_len, toks, tok_len, entry)
    """
    feats, feat_lens, toks, tok_lens, entries = zip(*batch)

    # if DEBUG_COLLATE:
    #     for i, e in enumerate(entries):
    #         print(f"[collate] sample {i}: utt_id={e.utt_id}, audio={e.audio_path}")

    feats     = pad_sequence(feats, batch_first=True, padding_value=0)                   # [B, T_max, D]
    feat_lens = torch.LongTensor(feat_lens)                             # [B]
    toks      = pad_sequence(toks, batch_first=True, padding_value=0)    # [B, L_max]
    tok_lens  = torch.LongTensor(tok_lens)                               # [B]

    return feats, feat_lens, toks, tok_lens, entries

def get_dataloaders(cfg: FinetuneConfig):
    bs = cfg.training.batch_size
    train_ds = VivosDataset(cfg, "train")
    valid_ds = VivosDataset(cfg, "valid")

    if len(train_ds) == 0:
        print("❌ [DataLoader] Empty train dataset.")
        train_loader = None
    else:
        train_loader = DataLoader(
            train_ds,
            batch_size=bs,
            shuffle=cfg.training.shuffle,
            collate_fn=collate_fn
        )

    if len(valid_ds) == 0:
        print("❌ [DataLoader] Empty valid dataset.")
        valid_loader = None
    else:
        valid_loader = DataLoader(
            valid_ds,
            batch_size=bs,
            shuffle=False,
            collate_fn=collate_fn
        )

    return train_loader, valid_loader


def get_dataloader_for_test_split(cfg, split_name="test"):
    bs = cfg.training.batch_size

    train_ds = VivosDatasetDebug(cfg, "train")

    train_loader = DataLoader(
        train_ds,
        batch_size=bs,
        shuffle=False,
        collate_fn=collate_fn_debug
    )

    return train_loader




def get_dataloaders_smoke(cfg: FinetuneConfig, ratio: float = 0.01):
    from torch.utils.data import Subset

    bs = cfg.training.batch_size

    train_ds = VivosDataset(cfg, "train")
    valid_ds = VivosDataset(cfg, "valid")

    if len(train_ds) == 0:
        print("❌ [SmokeLoader] Empty train dataset.")
        train_loader = None
    else:
        train_subset_size = max(1, int(len(train_ds) * ratio))
        train_subset = Subset(train_ds, list(range(train_subset_size)))
        train_loader = DataLoader(
            train_subset,
            batch_size=bs,
            shuffle=cfg.training.shuffle,
            collate_fn=collate_fn,
        )

    if len(valid_ds) == 0:
        print("❌ [SmokeLoader] Empty valid dataset.")
        valid_loader = None
    else:
        # @NOTE: luôn sử dụng full valid set -> ko cần smoke ratio 
        valid_subset_size = max(1, int(len(valid_ds)))
        valid_subset = Subset(valid_ds, list(range(valid_subset_size)))
        valid_loader = DataLoader(
            valid_subset,
            batch_size=bs,
            shuffle=False,
            collate_fn=collate_fn,
        )

    return train_loader, valid_loader



def get_dataloaders_aug_inspect(cfg: FinetuneConfig):
    """
    Tạo DataLoader để inspect từng sample đã augment (không shuffle, batch_size=1)
    """
    train_ds = VivosDataset(cfg, "train")
    valid_ds = VivosDataset(cfg, "valid")

    if len(train_ds) == 0:
        print("❌ [InspectLoader] Empty train dataset.")
        train_loader = None
    else:
        train_loader = DataLoader(
            train_ds,
            batch_size=1,  # 👈 để dễ iterate từng sample
            shuffle=False, # 👈 đảm bảo cùng thứ tự như file meta
            collate_fn=collate_fn_entry
        )

    if len(valid_ds) == 0:
        print("❌ [InspectLoader] Empty valid dataset.")
        valid_loader = None
    else:
        valid_loader = DataLoader(
            valid_ds,
            batch_size=1,
            shuffle=False,
            collate_fn=collate_fn_entry
        )

    return train_loader, valid_loader


#########################################################################




class VivosDatasetDebug(Dataset):
    def __init__(self, cfg: FinetuneConfig, split: str):
        self.cfg = cfg
        meta_file = f"{split}_meta.json"

        print(f"{cfg.data}")
        if split == "train" and cfg.data.train_meta_file:
            meta_file = cfg.data.train_meta_file

        manifest_file = os.path.join(cfg.data.manifest_dir, meta_file)
        with open(manifest_file, 'r', encoding='utf-8') as f:
            entries = json.load(f)
        self.meta = entries
        self.tokenizer = cfg.tokenizer.tokenizer

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        entry = self.meta[idx]
        audio_dir = self.cfg.data.audio_dir or self.cfg.data.manifest_dir
        abs_path = os.path.join(audio_dir, entry["audio_path"])
        wav, sr = torchaudio.load(abs_path)

        if sr != self.cfg.data.sample_rate:
            wav = torchaudio.transforms.Resample(sr, self.cfg.data.sample_rate)(wav)

        if wav.dim() == 1:
            wav = wav.unsqueeze(0)
        if wav.abs().max() <= 1.0:
            wav = wav * 32768.0
        wav = wav.clamp(-32768, 32767)

        feats = torchaudio.compliance.kaldi.fbank(
            wav,
            num_mel_bins = self.cfg.data.num_mel_bins,
            frame_length = self.cfg.data.frame_length,
            frame_shift  = self.cfg.data.frame_shift,
            dither       = self.cfg.data.dither,
            energy_floor = self.cfg.data.energy_floor,
            sample_frequency = self.cfg.data.sample_rate
        )

        norm_text = normalize_vi(entry["text"])
        token_ids = self.tokenizer.tokenize(norm_text)
        toks = torch.LongTensor(token_ids)

        # Return thêm gold + pred_old
        gold_corrected = entry.get("gold_corrected", None)
        pred_old = entry.get("pred_old", None)

        return feats, feats.size(0), toks, len(token_ids), entry["utt_id"], gold_corrected, pred_old


def collate_fn_debug(batch):
    feats, feat_lens, toks, tok_lens, utt_ids, golds, preds = zip(*batch)

    feats     = pad_sequence(feats, batch_first=True, padding_value=0)
    feat_lens = torch.LongTensor(feat_lens)
    toks      = pad_sequence(toks, batch_first=True, padding_value=0)
    tok_lens  = torch.LongTensor(tok_lens)

    return feats, feat_lens, toks, tok_lens, utt_ids, golds, preds



def get_dataloaders_debug(cfg: FinetuneConfig):
    bs = cfg.training.batch_size

    train_ds = VivosDatasetDebug(cfg, "train")
    valid_ds = VivosDatasetDebug(cfg, "valid")

    train_loader = DataLoader(
        train_ds,
        batch_size=bs,
        shuffle=False,
        collate_fn=collate_fn_debug
    )

    valid_loader = DataLoader(
        valid_ds,
        batch_size=bs,
        shuffle=False,
        collate_fn=collate_fn_debug
    )

    return train_loader, valid_loader

