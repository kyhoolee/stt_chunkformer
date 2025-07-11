#!/usr/bin/env python3
import json
import random
import torch
import torchaudio
from pathlib import Path
from jiwer import wer, Compose, ToLowerCase, RemovePunctuation, RemoveMultipleSpaces, Strip
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

# Normalization pipeline cho jiwer
norm_transform = Compose([
    ToLowerCase(),
    RemovePunctuation(),
    RemoveMultipleSpaces(),
    Strip()
])

# 2. Inspect a single example + compute WER
def inspect_and_score(entry: MetadataEntry, cfg: DataConfig,
                      model=None, char_dict=None, device=None, args=None):
    print(f"\n=== Inspect {entry.utt_id} ===")
    wav_path = Path(entry.audio_path)
    waveform, sr = torchaudio.load(wav_path)
    feat = torch.load(Path(cfg.cache_dir)/"features"/f"{entry.utt_id}.pt")

    print(f"WAV   : shape={tuple(waveform.shape)}, sr={sr}")
    print(f"FEAT  : shape={tuple(feat.shape)}, dtype={feat.dtype}")

    if model and args:
        # Decode
        args.audio_path = str(wav_path)
        ctc_pred = decode_long_form(prepare_input_file(args.audio_path, device),
                                    model, char_dict, args, device)
        # Normalize GT and prediction
        gt = norm_transform(entry.text)
        pred = norm_transform(ctc_pred)
        # Compute WER
        sample_wer = wer(gt, pred)
        print(f"GT    : {gt}")
        print(f"PRED  : {pred}")
        print(f"WER   : {sample_wer:.3f}")
        return sample_wer
    return None

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
    args.model_checkpoint = model_checkpoint

    # 4. Load metadata and sample
    metadata = load_metadata(cfg.cache_dir, split="train")
    k = min(1000, len(metadata))
    samples = random.sample(metadata, k=k)

    # 5. Inspect each and collect WERs
    wers = []
    for entry in samples:
        sample_wer = inspect_and_score(entry, cfg,
                                       model=model,
                                       char_dict=char_dict,
                                       device=device,
                                       args=args)
        if sample_wer is not None:
            wers.append(sample_wer)

    # 6. Print average WER
    if wers:
        avg_wer = sum(wers) / len(wers)
        print(f"\n► Average WER over {len(wers)} samples: {avg_wer:.3f}")

if __name__ == "__main__":
    main()
