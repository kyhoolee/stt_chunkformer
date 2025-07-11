#!/usr/bin/env python3
import json
import random
import torch
import torchaudio
from pathlib import Path
from chunkformer_vpb.data.data import DataConfig, MetadataEntry, compute_fbank
from chunkformer_vpb.model_utils import (
    init, prepare_input_file, decode_long_form,
    decode_aed_long_form, get_default_args
)

# 1. Load metadata for a split
def load_metadata(cache_dir: str, split: str = "test") -> list[MetadataEntry]:
    meta_path = Path(cache_dir) / f"{split}_meta.json"
    with open(meta_path, "r", encoding="utf-8") as f:
        entries = json.load(f)
    return [MetadataEntry(**e) for e in entries]

# 2. Inspect a single example
def inspect_example(entry: MetadataEntry, cfg: DataConfig,
                    model=None, char_dict=None, device=None, args=None):
    print(f"\n=== Inspect {entry.utt_id} ===")
    wav_path = Path(entry.audio_path)

    # a) check WAV
    waveform, sr = torchaudio.load(wav_path)
    print(f"WAV  : shape={tuple(waveform.shape)}, sr={sr}, "
          f"min={waveform.min():.2f}, max={waveform.max():.2f}")

    # b) check saved feature
    feat_path = Path(cfg.cache_dir) / "features" / f"{entry.utt_id}.pt"
    feat = torch.load(feat_path)
    print(f"Feat : shape={tuple(feat.shape)}, dtype={feat.dtype}, "
          f"min={feat.min():.4f}, max={feat.max():.4f}")

    # c) recompute & compare
    recompute = compute_fbank(waveform, cfg)
    diff = (recompute - feat).abs().mean().item()
    print(f"Recompute fbank diff mean: {diff:.6f}")

    # d) inference if model + args provided
    if model and args:
        args.audio_path = str(wav_path)
        feats = prepare_input_file(args.audio_path, device)
        ctc = decode_long_form(feats, model, char_dict, args, device)
        aed_raw, aed_clean = decode_aed_long_form(feats, model, char_dict, args, device)
        print(f"CTC : {ctc}")
        print(f"AED : {aed_clean}")
        print(f"GT  : {entry.text}")

def main():
    # 1. Config pipeline
    cfg = DataConfig(
        cache_dir="./cache_test",
        dataset_name="AILAB-VNUHCM/vivos",
        dataset_split="train",
        vocab_path="./vocab.txt"
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2. Load model
    model_checkpoint = "../../../chunkformer-large-vie"
    model, char_dict = init(model_checkpoint, device)
    model.eval()

    # 3. Prepare inference args
    args = get_default_args()
    # optionally override any defaults:
    args.model_checkpoint = model_checkpoint

    # 4. Load metadata and sample
    metadata = load_metadata(cfg.cache_dir, split="train")
    n = len(metadata)
    k = min(5, n)
    samples = random.sample(metadata, k=k)

    # 5. Inspect each
    for entry in samples:
        inspect_example(entry, cfg, model=model,
                        char_dict=char_dict, device=device, args=args)

if __name__ == "__main__":
    main()
