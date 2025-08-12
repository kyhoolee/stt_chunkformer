import os
import json
from pathlib import Path
from collections import defaultdict

def check_split_manifest(dataset_name: str, manifest_dir: Path, audio_root: Path):
    print(f"\n📁 Dataset: {dataset_name}")

    for split in ["train", "dev", "test"]:
        manifest_path = manifest_dir / f"{dataset_name}_{split}_manifest.json"
        audio_dir = audio_root / dataset_name / split / "audio"

        if not manifest_path.exists():
            print(f"  ❌ No manifest for {split}")
            continue

        if not audio_dir.exists():
            print(f"  ❌ No audio folder for {split}")
            continue

        # Load manifest entries (basename only)
        with open(manifest_path, "r") as f:
            lines = f.readlines()

        manifest_files = set()
        for line in lines:
            try:
                entry = json.loads(line)
                wav_path = entry.get("wav", "")
                if wav_path:
                    basename = os.path.basename(wav_path)
                    manifest_files.add(basename)
            except Exception as e:
                print(f"  ⚠️ Error reading manifest line: {e}")

        # Get actual .wav files
        actual_files = set(f.name for f in audio_dir.glob("*.wav"))

        # Compare
        missing = manifest_files - actual_files
        extra = actual_files - manifest_files

        print(f"  ✅ {split:<5}: {len(manifest_files)} in manifest | {len(actual_files)} in folder")
        if missing:
            print(f"     ❗ Missing files in folder: {len(missing)}")
            for f in list(missing)[:5]:
                print(f"       - {f}")
        if extra:
            print(f"     ⚠️ Extra files in folder: {len(extra)}")
            for f in list(extra)[:5]:
                print(f"       - {f}")

def main():
    datasets = [
        "vi_voice", "viet_bud500", "vietspeech"  # những tập đã có split manifest rõ ràng
    ]

    manifest_root = Path(os.path.expanduser("~/dataset/data/manifests"))
    audio_root = Path(os.path.expanduser("~/dataset/data/data/processed/8khz"))

    print("📂 Verifying manifest file name alignment...\n")
    for ds in datasets:
        check_split_manifest(ds, manifest_root, audio_root)

if __name__ == "__main__":
    main()
