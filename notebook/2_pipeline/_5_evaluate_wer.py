#!/usr/bin/env python3
import json
import torch
import torchaudio
import statistics
from pathlib import Path
from jiwer import wer, Compose, ToLowerCase, RemovePunctuation, RemoveMultipleSpaces, Strip
from chunkformer_vpb.data.data import DataConfig, MetadataEntry
from chunkformer_vpb.model_utils import init, prepare_input_file, decode_long_form, get_default_args

# Normalization pipeline for WER
norm_transform = Compose([
    ToLowerCase(),
    RemovePunctuation(),
    RemoveMultipleSpaces(),
    Strip()
])

def load_metadata(cache_dir: str, split: str = "test") -> list[MetadataEntry]:
    meta_path = Path(cache_dir) / f"{split}_meta.json"
    with open(meta_path, "r", encoding="utf-8") as f:
        entries = json.load(f)
    return [MetadataEntry(**e) for e in entries]

def compute_wers_on_dataset(cfg: DataConfig,
                            split: str = "test",
                            args=None,
                            device=None) -> None:
    # 1. load metadata
    metadata = load_metadata(cfg.cache_dir, split)
    if not metadata:
        print(f"No entries in split '{split}'.")
        return

    # 2. load model
    model, char_dict = init(args.model_checkpoint, device)
    model.eval()

    wers = []
    for entry in metadata:
        # a) decode CTC
        feats = prepare_input_file(str(entry.audio_path), device)
        pred = decode_long_form(feats, model, char_dict, args, device)

        # b) normalize
        gt_norm   = norm_transform(entry.text)
        pred_norm = norm_transform(pred)

        # c) compute WER
        sample_wer = wer(gt_norm, pred_norm)
        wers.append(sample_wer)

    # 3. compute statistics
    avg_wer  = statistics.mean(wers)
    min_wer  = min(wers)
    max_wer  = max(wers)
    median   = statistics.median(wers)
    p25      = statistics.quantiles(wers, n=4)[0]   # first quartile
    p75      = statistics.quantiles(wers, n=4)[2]   # third quartile
    p95      = statistics.quantiles(wers, n=100)[94]  # 95th percentile

    # 4. report
    print(f"Dataset: {cfg.dataset_name}, split: {split}")
    print(f"Entries evaluated: {len(wers)}")
    print(f"Average WER : {avg_wer:.3f}")
    print(f"Min WER     : {min_wer:.3f}")
    print(f"Max WER     : {max_wer:.3f}")
    print(f"Median WER  : {median:.3f}")
    print(f"25th pctile : {p25:.3f}")
    print(f"75th pctile : {p75:.3f}")
    print(f"95th pctile : {p95:.3f}")

if __name__ == "__main__":
    # configure
    cfg = DataConfig(
        cache_dir="./cache_test",
        dataset_name="AILAB-VNUHCM/vivos",
        dataset_split="test",
        vocab_path="./vocab.txt"
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # args for decoding
    args = get_default_args()
    args.model_checkpoint = "../../../chunkformer-large-vie"

    # run
    compute_wers_on_dataset(cfg, split="test", args=args, device=device)
