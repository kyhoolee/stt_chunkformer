import os
import json
from pathlib import Path

# Danh sách dataset chuẩn từ file .tar
expected_datasets = [
    "fpt_fosd", "infore", "lsvsc", "speech_massive", "vais1000",
    "vi_voice", "viet_bud500", "vietmed", "vivos", "vlsp2020", "vietspeech"
]

# Root path
DATA_ROOT = Path(os.path.expanduser("~/dataset/data/data/processed/8khz"))
MANIFEST_ROOT = Path(os.path.expanduser("~/dataset/data/manifests"))

def load_manifest_count(dataset_name, split):
    """
    Trả về số lượng mẫu trong manifest json nếu tồn tại
    """
    manifest_file = MANIFEST_ROOT / f"{dataset_name}_{split}_manifest.json"
    if not manifest_file.exists():
        return None
    try:
        with open(manifest_file) as f:
            lines = f.readlines()
        return len(lines)
    except:
        return None

def check_datasets(data_root: Path, expected: list):
    found_datasets = [d.name for d in data_root.iterdir() if d.is_dir()]
    report = {}

    missing = list(set(expected) - set(found_datasets))
    extra = list(set(found_datasets) - set(expected))

    print(f"✅ Found {len(found_datasets)} datasets.")
    if missing:
        print(f"❌ Missing datasets: {missing}")
    if extra:
        print(f"⚠️ Extra datasets: {extra}")

    for ds in expected:
        ds_path = data_root / ds
        ds_report = {}
        for split in ['train', 'dev', 'test']:
            split_path = ds_path / split / 'audio'
            if not split_path.exists():
                file_count = 0
                exists = False
            else:
                file_count = len(list(split_path.glob("*.wav")))
                exists = True

            manifest_count = load_manifest_count(ds, split)
            ds_report[split] = {
                'exists': exists,
                'num_wav': file_count,
                'manifest_samples': manifest_count
            }
        report[ds] = ds_report

    return report

def print_report(report):
    print("\n📊 Dataset Summary (File vs Manifest):")
    for ds, stats in report.items():
        print(f"\n📁 {ds}:")
        for split, info in stats.items():
            status = "✅" if info['exists'] else "❌ MISSING"
            wav_count = info['num_wav']
            manifest_count = info['manifest_samples']
            manifest_str = (
                f"{manifest_count} samples"
                if manifest_count is not None else "no manifest"
            )
            match = "✅ MATCH" if manifest_count == wav_count else "❌ MISMATCH"
            print(f"  {split:<5}: {status} - {wav_count} .wav files | {manifest_str} | {match}")

if __name__ == "__main__":
    result = check_datasets(DATA_ROOT, expected_datasets)
    print_report(result)
