import os
import json
from pathlib import Path

def inspect_manifest_format(dataset_name: str, manifest_dir: Path, audio_root: Path, max_preview=3):
    """
    Đọc manifest gốc (không chia theo split), so sánh với tổng số .wav file trong folder train/dev/test.
    """
    manifest_path = manifest_dir / f"{dataset_name}_manifest.json"
    if not manifest_path.exists():
        print(f"\n❌ Manifest not found for {dataset_name}: {manifest_path}")
        return
    
    with open(manifest_path, "r") as f:
        lines = f.readlines()
    
    print(f"\n📁 Dataset: {dataset_name}")
    print(f"📑 Manifest entries       : {len(lines)}")

    # Tổng số file thực tế trong cả 3 split
    audio_count = 0
    for split in ['train', 'dev', 'test']:
        audio_dir = audio_root / dataset_name / split / 'audio'
        if audio_dir.exists():
            count = len(list(audio_dir.glob("*.wav")))
            audio_count += count
    print(f"🎧 Actual .wav files total: {audio_count}")

    # So sánh
    match = "✅ MATCH" if audio_count == len(lines) else "❌ MISMATCH"
    print(f"🔍 Compare result         : {match}")

    # In preview sample
    print("\n🔎 Manifest preview:")
    for line in lines[:max_preview]:
        try:
            sample = json.loads(line)
            print(json.dumps(sample, indent=2))
        except Exception as e:
            print(f"⚠️ Error decoding line: {e}")

def main():
    datasets_to_check = [
        "fpt_fosd", "infore", "lsvsc", "speech_massive", "vais1000",
        "vietmed", "vivos", "vlsp2020"
    ]

    manifest_root = Path(os.path.expanduser("~/dataset/data/manifests"))
    audio_root = Path(os.path.expanduser("~/dataset/data/data/processed/8khz"))

    print("📂 Checking dataset manifests...\n")
    for ds in datasets_to_check:
        inspect_manifest_format(ds, manifest_root, audio_root)

if __name__ == "__main__":
    main()
