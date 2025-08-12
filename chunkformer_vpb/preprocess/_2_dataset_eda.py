import os
import random
import soundfile as sf
from pathlib import Path

def check_sample_rate(dataset_root, n_sample=3):
    stats = {}
    for dataset_dir in Path(dataset_root).iterdir():
        if dataset_dir.is_dir():
            stats[dataset_dir.name] = {}
            for split in ['train', 'dev', 'test']:
                audio_dir = dataset_dir / split / 'audio'
                if not audio_dir.exists():
                    continue
                files = list(audio_dir.glob("*.wav"))
                sample_files = random.sample(files, min(n_sample, len(files)))
                rates = []
                for f in sample_files:
                    rate = sf.info(str(f)).samplerate
                    rates.append(rate)
                stats[dataset_dir.name][split] = {
                    'num_files': len(files),
                    'sample_rates': rates,
                }
    return stats


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Check sample rates in dataset")
    parser.add_argument("dataset_root", type=str, help="Root directory of the dataset")
    parser.add_argument("--n_sample", type=int, default=3, help="Number of samples to check per split")
    args = parser.parse_args()

    stats = check_sample_rate(args.dataset_root, n_sample=args.n_sample)
    for dataset, splits in stats.items():
        print(f"Dataset: {dataset}")
        for split, info in splits.items():
            print(f"  Split: {split}, Num files: {info['num_files']}, Sample rates: {info['sample_rates']}")

# python -m work.stt_chunkformer.chunkformer_vpb.preprocess._2_dataset_eda --dataset_root /path/to/dataset --n_sample 5