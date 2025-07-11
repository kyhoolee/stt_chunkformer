import os
import json
import yaml
import shutil
import time
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
import torch
from torch.utils.data import Dataset
from multiprocessing import Pool
from datasets import load_dataset
from pydub import AudioSegment
import torchaudio
import logging
from tqdm import tqdm


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
    Load audio robustly: first try pydub, fallback to torchaudio.load,
    ensure 16 kHz mono PCM 16-bit scaled in float32.
    Returns waveform tensor shape (1, N), dtype float32 with values in int16 range.
    """
    try:
        audio = AudioSegment.from_file(audio_path)
        audio = audio.set_frame_rate(16000).set_sample_width(2).set_channels(1)
        raw = audio.get_array_of_samples()  # int16 values
        waveform = torch.as_tensor(raw, dtype=torch.float32).unsqueeze(0)
        logging.debug(f"[pydub] Loaded {audio_path}: shape={waveform.shape}, "
                      f"min={waveform.min().item():.1f}, max={waveform.max().item():.1f}")
    except Exception as e:
        # log full exception
        logging.warning(f"[pydub] Failed for {audio_path} with: {e!r}")
        waveform, sr = torchaudio.load(audio_path)
        if waveform.dim() == 2:
            waveform = waveform.mean(dim=0, keepdim=True)
        if sr != 16000:
            waveform = torchaudio.transforms.Resample(sr, 16000)(waveform)
        # scale normalized float to int16 range
        if waveform.abs().max() <= 1.0:
            waveform = waveform * 32768.0
        waveform = waveform.clamp(-32768, 32767)
        logging.debug(f"[torchaudio] Loaded {audio_path}: shape={waveform.shape}, "
                      f"min={waveform.min().item():.1f}, max={waveform.max().item():.1f}")
    return waveform


def augment_and_extract(args: Tuple[MetadataEntry, DataConfig]) -> Optional[MetadataEntry]:
    entry, config = args
    try:
        # logging.info(f"→ [{entry.utt_id}] Start processing")
        waveform = load_audio(entry.audio_path)
        # logging.info(f"   waveform stats: min={waveform.min().item():.1f}, "
        #              f"max={waveform.max().item():.1f}, mean={waveform.mean().item():.1f}")
        feat = compute_fbank(waveform, config)

        os.makedirs(Path(config.cache_dir) / 'features', exist_ok=True)
        # Save the feature tensor to a file
        out_path = Path(config.cache_dir) / 'features' / f"{entry.utt_id}.pt"
        torch.save(feat, out_path)
        # logging.info(f"→ [{entry.utt_id}] Saved feature ({feat.shape})")
        return entry
    except Exception:
        logging.exception(f"✗ [{entry.utt_id}] Error in processing")
        return None


def compute_fbank(waveform: torch.Tensor, config: DataConfig) -> torch.Tensor:
    feat = torchaudio.compliance.kaldi.fbank(
        waveform,
        num_mel_bins=config.fbank['n_mels'],
        frame_length=config.fbank['win_length']/16000*1000,
        frame_shift=config.fbank['hop_length']/16000*1000,
        dither=0.0,
        energy_floor=0.0,
        sample_frequency=16000
    )
    return feat

def save_waveform(wav_path: Path, waveform: torch.Tensor, sr: int):
    """
    Save a waveform tensor to WAV as PCM 16-bit,
    letting torchaudio handle any necessary conversion.
    """
    torchaudio.save(
        str(wav_path),
        waveform,            # float32 or int16 tensor
        sr,
        encoding="PCM_S",    # signed integer PCM
        bits_per_sample=16
    )
    logging.debug(f"→ Saved WAV: {wav_path} (shape={waveform.shape}, sr={sr})")


def extract_hf_dataset(config: DataConfig) -> List[MetadataEntry]:
    logging.info(f"→ Loading HF dataset {config.dataset_name}:{config.dataset_split}")
    ds = load_dataset(
        config.dataset_name,
        split=config.dataset_split,
        trust_remote_code=True
    )
    raw_dir = Path(config.cache_dir) / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    metadata: List[MetadataEntry] = []
    for i, example in enumerate(ds):
        utt_id   = f"utt_{i:06d}"
        wav_path = raw_dir / f"{utt_id}.wav"

        if wav_path.exists():
            logging.info(f"[{utt_id}] Cached WAV found, skipping save")
        else:
            logging.info(f"[{utt_id}] Saving WAV via torchaudio.save()")
            audio_dict = example["audio"]
            # HF provides normalized float32 array [-1,1]
            waveform = torch.tensor(audio_dict["array"], dtype=torch.float32).unsqueeze(0)
            sr       = audio_dict["sampling_rate"]
            # torchaudio.save will handle scaling to PCM_16
            save_waveform(wav_path, waveform, sr)

        metadata.append(
            MetadataEntry(utt_id=utt_id,
                          text=example["sentence"],
                          audio_path=str(wav_path))
        )

    logging.info(f"→ Extracted {len(metadata)} examples")
    return metadata



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


def run_pipeline(config_path: str,
                 test: bool = False,
                 test_size: int = 10,
                 use_progress: bool = True,
                 save_meta_in_test: bool = True):
    start_time = time.time()
    logging.info(f"Starting pipeline (test={test}, size={test_size})")
    cfg = DataConfig(**yaml.safe_load(open(config_path)))
    # 1. Extract & split metadata
    metadata = extract_hf_dataset(cfg)
    metadata = split_metadata(metadata, cfg)

    # 2. Giới hạn nếu test
    if test:
        metadata = metadata[:test_size]

    total = len(metadata)
    logging.info(f"Total tasks to process: {total}")

    # 3. Xử lý feature
    tasks = [(e, cfg) for e in metadata]
    if use_progress:
        iterator = tqdm(tasks, desc="Tasks", ncols=80)
        results = [augment_and_extract(args) for args in iterator]
    else:
        with Pool(min(8, os.cpu_count())) as pool:
            results = pool.map(augment_and_extract, tasks)

    success = sum(1 for r in results if r is not None)
    duration = time.time() - start_time
    logging.info(f"Pipeline completed: {success}/{total} succeeded in {duration:.2f}s")

    # 4. Luôn lưu meta—nếu test chỉ tạo meta cho subset
    if not test or save_meta_in_test:
        # rebuild splits only on this subset
        # (nhóm lại theo split của từng entry trong metadata)
        for split in ['train', 'valid', 'test']:
            entries = [e.__dict__ for e in metadata if e.split == split]
            out_file = Path(cfg.cache_dir) / f"{split}_meta.json"
            out_file.write_text(json.dumps(entries, ensure_ascii=False, indent=2))
            logging.info(f"Saved metadata for {split} to {out_file}")

    return {'total': total, 'success': success, 'duration': duration}


# CLI support
def build_data(config_path: str):
    return run_pipeline(config_path, test=False, use_progress=False)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config.yaml')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--test_size', type=int, default=10)
    parser.add_argument('--progress', action='store_true')
    args = parser.parse_args()
    run_pipeline(args.config, test=args.test, test_size=args.test_size, use_progress=args.progress)
