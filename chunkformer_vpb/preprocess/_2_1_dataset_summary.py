import os
from pathlib import Path
from collections import defaultdict

# Danh sách dataset chuẩn từ file .tar
expected_datasets = [
    "fpt_fosd", "infore", "lsvsc", "speech_massive", "vais1000",
    "vi_voice", "viet_bud500", "vietmed", "vivos", "vlsp2020",
    'vietspeech',
]

# Root path đến folder chứa dataset
DATA_ROOT = Path(os.path.expanduser("~/dataset/data/data/processed/8khz"))


def check_datasets(data_root: Path, expected: list):
    found_datasets = [d.name for d in data_root.iterdir() if d.is_dir()]
    report = {}

    # Kiểm tra thiếu hoặc thừa
    missing = list(set(expected) - set(found_datasets))
    extra = list(set(found_datasets) - set(expected))

    print(f"✅ Found {len(found_datasets)} datasets.")
    if missing:
        print(f"❌ Missing datasets: {missing}")
    if extra:
        print(f"⚠️ Extra datasets: {extra}")

    # Thống kê chi tiết từng dataset
    for ds in expected:
        ds_path = data_root / ds
        ds_report = {}
        for split in ['train', 'dev', 'test']:
            split_path = ds_path / split / 'audio'
            if not split_path.exists():
                ds_report[split] = {'exists': False, 'num_wav': 0}
            else:
                num_wav = len(list(split_path.glob("*.wav")))
                ds_report[split] = {'exists': True, 'num_wav': num_wav}
        report[ds] = ds_report

    return report

def print_report(report):
    print("\n📊 Dataset Summary:")
    for ds, stats in report.items():
        print(f"\n📁 {ds}:")
        for split, info in stats.items():
            status = "✅" if info['exists'] else "❌ MISSING"
            print(f"  {split:<5}: {status} - {info['num_wav']} .wav files")

if __name__ == "__main__":
    result = check_datasets(DATA_ROOT, expected_datasets)
    print_report(result)
