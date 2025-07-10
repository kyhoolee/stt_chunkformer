import os
import json
import yaml
import shutil
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Tuple
import torch
from torch.utils.data import Dataset
from multiprocessing import Pool
from datasets import load_dataset
from pydub import AudioSegment
import torchaudio
import logging

# Configure logging
logging.basicConfig(
    format='[%(asctime)s] %(levelname)s: %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S'
)

@dataclass
class DataConfig:
    cache_dir: str
    dataset_name: str  # e.g. "AILAB-VNUHCM/vivos"
    dataset_split: str = 'train'
    vocab_path: str = ''
    splits: Dict[str, float] = field(default_factory=lambda: {"train": 0.8, "valid": 0.1, "test": 0.1})
    augment: Dict = field(default_factory=lambda: {
        "speed": [0.9, 1.0, 1.1],
        "freq_mask": {"num_masks": 2, "param": 15},
        "noise_types": ["telephony", "vietnamese"],
    })
    fbank: Dict = field(default_factory=lambda: {"n_mels": 80, "win_length": 400, "hop_length": 160})
    seed: int = 42

@dataclass
class MetadataEntry:
    utt_id: str
    text: str
    audio_path: str
    split: str = ''


def load_audio(audio_path: str) -> torch.Tensor:
    """
    Load and normalize audio via pydub into raw PCM 16-bit values, shape (1, N), dtype float32.
    """
    logging.info(f"Loading audio: {audio_path}")
    audio = AudioSegment.from_file(audio_path)
    logging.debug(f"Original frame_rate={audio.frame_rate}, sample_width={audio.sample_width}, channels={audio.channels}")
    audio = audio.set_frame_rate(16000).set_sample_width(2).set_channels(1)
    raw = audio.get_array_of_samples()
    waveform = torch.as_tensor(raw, dtype=torch.float32).unsqueeze(0)
    logging.debug(f"Waveform shape after pydub load: {waveform.shape}")
    return waveform


def compute_fbank(waveform: torch.Tensor, config: DataConfig) -> torch.Tensor:
    """
    Compute log-Mel-filterbank from raw PCM waveform.
    """
    logging.debug("Computing fbank features")
    feat = torchaudio.compliance.kaldi.fbank(
        waveform,
        num_mel_bins=config.fbank['n_mels'],
        frame_length=config.fbank['win_length']/16000*1000,
        frame_shift=config.fbank['hop_length']/16000*1000,
        dither=0.0,
        energy_floor=0.0,
        sample_frequency=16000
    )
    logging.debug(f"Feature shape: {feat.shape}")
    return feat


def extract_hf_dataset(config: DataConfig) -> List[MetadataEntry]:
    logging.info(f"Extracting HuggingFace dataset {config.dataset_name}:{config.dataset_split}")
    ds = load_dataset(config.dataset_name, split=config.dataset_split)
    raw_dir = Path(config.cache_dir) / 'raw'
    raw_dir.mkdir(parents=True, exist_ok=True)
    metadata: List[MetadataEntry] = []
    for i, example in enumerate(ds):
        utt_id = f"utt_{i:06d}"
        wav_path = raw_dir / f"{utt_id}.wav"
        # load audio array and sr directly
        audio_dict = example['audio']  # contains 'array' and 'sampling_rate'
        waveform = torch.tensor(audio_dict['array'], dtype=torch.float32).unsqueeze(0)
        sr = audio_dict['sampling_rate']
        # save wav
        torchaudio.save(wav_path, waveform, sr)
        logging.debug(f"Saved raw audio to {wav_path}")
        metadata.append(MetadataEntry(utt_id=utt_id, text=example['sentence'], audio_path=str(wav_path)))
    logging.info(f"Extracted {len(metadata)} examples")
    return metadata


def augment_and_extract(args: Tuple[MetadataEntry, DataConfig]) -> MetadataEntry:
    entry, config = args
    feat_dir = Path(config.cache_dir) / 'features'
    feat_dir.mkdir(parents=True, exist_ok=True)

    logging.info(f"Processing {entry.utt_id}")
    # 1. load raw PCM waveform via pydub loader
    waveform = load_audio(entry.audio_path)

    # 2. (optional) apply speed perturb, noise, freq masking here on PCM tensor
    # TODO: implement augmentations on waveform

    # 3. compute fbank
    feat = compute_fbank(waveform, config)

    # 4. save feature
    out_path = feat_dir / f"{entry.utt_id}.pt"
    torch.save(feat, out_path)
    logging.debug(f"Saved feature to {out_path}")
    return entry


def split_metadata(metadata: List[MetadataEntry], config: DataConfig) -> List[MetadataEntry]:
    import random
    logging.info("Splitting metadata into train/valid/test")
    random.seed(config.seed)
    utts = metadata.copy()
    random.shuffle(utts)
    n = len(utts)
    train_end = int(config.splits['train'] * n)
    valid_end = int((config.splits['train'] + config.splits['valid']) * n)
    for idx, entry in enumerate(utts):
        if idx < train_end:
            entry.split = 'train'
        elif idx < valid_end:
            entry.split = 'valid'
        else:
            entry.split = 'test'
    counts = {s: sum(1 for e in utts if e.split == s) for s in ['train','valid','test']}
    logging.info(f"Split counts: {counts}")
    return utts

class VivosDataset(Dataset):
    def __init__(self, entries: List[MetadataEntry], split: str,
                 config: DataConfig, text_transform):
        self.entries = [e for e in entries if e.split == split]
        self.config = config
        self.tt = text_transform
        logging.info(f"Initialized {split} dataset with {len(self.entries)} examples")

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        e = self.entries[idx]
        feat_path = Path(self.config.cache_dir) / 'features' / f"{e.utt_id}.pt"
        logging.debug(f"Loading feature {feat_path}")
        feat = torch.load(feat_path)
        label = torch.tensor(self.tt.text_to_ids(e.text), dtype=torch.long)
        return {'input': feat, 'label': label}


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    logging.debug("Collating batch")
    inputs = [b['input'] for b in batch]
    input_lengths = torch.tensor([i.size(0) for i in inputs], dtype=torch.long)
    padded_inputs = torch.nn.utils.rnn.pad_sequence(inputs, batch_first=True)

    labels = [b['label'] for b in batch]
    label_lengths = torch.tensor([l.size(0) for l in labels], dtype=torch.long)
    padded_labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=0)

    return {
        'inputs': padded_inputs,
        'input_lengths': input_lengths,
        'labels': padded_labels,
        'label_lengths': label_lengths,
    }


def build_data(config_path: str):
    logging.info(f"Loading config from {config_path}")
    cfg = DataConfig(**yaml.safe_load(open(config_path)))
    metadata = extract_hf_dataset(cfg)
    metadata = split_metadata(metadata, cfg)
    logging.info("Starting parallel feature extraction")
    with Pool(min(8, os.cpu_count())) as pool:
        pool.map(augment_and_extract, [(e, cfg) for e in metadata])
    for split in ['train', 'valid', 'test']:
        entries = [e.__dict__ for e in metadata if e.split == split]
        out_file = Path(cfg.cache_dir) / f"{split}_meta.json"
        out_file.write_text(json.dumps(entries, ensure_ascii=False, indent=2))
        logging.info(f"Saved metadata for {split} to {out_file}")
    logging.info("Data pipeline finished.")
