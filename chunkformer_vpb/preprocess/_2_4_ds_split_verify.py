import os
import json
from pathlib import Path
from collections import defaultdict

def check_manifest_and_wav(dataset_name: str, manifest_dir: Path, audio_root: Path):
    manifest_path = manifest_dir / f"{dataset_name}_manifest.json"
    if not manifest_path.exists():
        print(f"\n❌ Manifest not found for {dataset_name}")
        return

    with open(manifest_path, "r") as f:
        lines = f.readlines()

    # Phân nhóm theo split
    split_counts = defaultdict(int)
    split_wav_files = defaultdict(list)

    for line in lines:
        try:
            sample = json.loads(line)
            split = sample.get("split", "unknown")
            wav_file = os.path.basename(sample.get("wav", ""))
            split_counts[split] += 1
            split_wav_files[split].append(wav_file)
        except Exception as e:
            print(f"⚠️ Error parsing line in manifest: {e}")
            continue

    print(f"\n📁 Dataset: {dataset_name}")
    print(f"📑 Total manifest entries: {len(lines)}")
    print("🔢 Sample count by split:")
    for split, count in split_counts.items():
        print(f"  - {split:<5}: {count} samples")

    # Kiểm tra tồn tại thực tế các file .wav trong folder tương ứng
    print("\n🔍 Verifying .wav file existence by split:")
    for split in ['train', 'dev', 'test']:
        audio_dir = audio_root / dataset_name / split / 'audio'
        if not audio_dir.exists():
            print(f"  ❌ {split:<5}: audio folder missing")
            continue

        actual_wavs = {f.name for f in audio_dir.glob("*.wav")}
        manifest_wavs = set(split_wav_files.get(split, []))
        missing = manifest_wavs - actual_wavs
        extra = actual_wavs - manifest_wavs

        print(f"  ✅ {split:<5}: {len(manifest_wavs)} in manifest | {len(actual_wavs)} in folder")
        if missing:
            print(f"     ❗ Missing in folder: {len(missing)} files")
        if extra:
            print(f"     ⚠️ Extra in folder:   {len(extra)} files")

def main():
    datasets_to_check = [
        "fpt_fosd", "infore", "lsvsc", "speech_massive", "vais1000",
        "vietmed", "vivos", "vlsp2020"
    ]

    manifest_root = Path(os.path.expanduser("~/dataset/data/manifests"))
    audio_root = Path(os.path.expanduser("~/dataset/data/data/processed/8khz"))

    print("📂 Checking dataset manifest consistency and wav alignment...\n")
    for ds in datasets_to_check:
        check_manifest_and_wav(ds, manifest_root, audio_root)

if __name__ == "__main__":
    main()
