import os
import json
import soundfile as sf
from pathlib import Path
from collections import defaultdict

# 🔹 Nhóm 1: manifest không chia split
datasets_unsplit = [
    "fpt_fosd", "infore", "lsvsc", "speech_massive", "vais1000",
    "vietmed", "vivos", "vlsp2020"
]

# 🔹 Nhóm 2: manifest chia rõ train/dev/test
datasets_split = [
    "vi_voice", "viet_bud500", "vietspeech"
]

manifest_root = Path(os.path.expanduser("~/dataset/data/manifests"))
audio_root = Path(os.path.expanduser("~/dataset/data/data/processed/8khz"))

def check_sample_rate(file_path: Path, expected_rate: int = 8000):
    try:
        info = sf.info(str(file_path))
        return info.samplerate == expected_rate, info.samplerate
    except Exception as e:
        return False, f"ERROR: {e}"

def check_split_manifest(dataset: str, split: str, manifest_path: Path):
    audio_dir = audio_root / dataset / split / "audio"

    if not audio_dir.exists():
        print(f"  ❌ Audio folder not found: {audio_dir}")
        return

    with open(manifest_path, "r") as f:
        lines = f.readlines()

    missing_files = []
    wrong_sr_files = []

    for line in lines:
        try:
            entry = json.loads(line)
            wav_path = entry.get("wav", "")
            wav_name = os.path.basename(wav_path)
            audio_path = audio_dir / wav_name

            if not audio_path.exists():
                missing_files.append(wav_name)
                continue

            is_ok, sr = check_sample_rate(audio_path)
            if not is_ok:
                wrong_sr_files.append((wav_name, sr))
        except Exception as e:
            print(f"  ⚠️ JSON error: {e}")

    print(f"  ✅ {split:<5}: {len(lines)} samples checked")
    if missing_files:
        print(f"     ❗ Missing: {len(missing_files)} files (e.g., {missing_files[:3]})")
    if wrong_sr_files:
        print(f"     ❌ Wrong SR: {len(wrong_sr_files)} files (e.g., {wrong_sr_files[:3]})")
    if not missing_files and not wrong_sr_files:
        print(f"     👍 All files OK (8kHz)")

def check_unsplit_manifest(dataset: str, manifest_path: Path):
    with open(manifest_path, "r") as f:
        lines = f.readlines()

    missing_by_split = defaultdict(list)
    wrong_sr_by_split = defaultdict(list)

    for line in lines:
        try:
            entry = json.loads(line)
            split = entry.get("split", "unknown")
            wav_path = entry.get("wav", "")
            wav_name = os.path.basename(wav_path)
            audio_path = audio_root / dataset / split / "audio" / wav_name

            if not audio_path.exists():
                missing_by_split[split].append(wav_name)
                continue

            is_ok, sr = check_sample_rate(audio_path)
            if not is_ok:
                wrong_sr_by_split[split].append((wav_name, sr))
        except Exception as e:
            print(f"  ⚠️ JSON error: {e}")

    print(f"📁 {dataset}: {len(lines)} samples")

    for split in sorted(set(list(missing_by_split) + list(wrong_sr_by_split))):
        m = missing_by_split.get(split, [])
        w = wrong_sr_by_split.get(split, [])
        print(f"  🔸 Split: {split}")
        if m:
            print(f"     ❗ Missing: {len(m)} files (e.g., {m[:3]})")
        if w:
            print(f"     ❌ Wrong SR: {len(w)} files (e.g., {w[:3]})")
        if not m and not w:
            print(f"     👍 All files OK (8kHz)")

def main():
    print("🎧 Checking sample rate: 8kHz expectation\n")

    # Nhóm 1: manifest gộp
    for dataset in datasets_unsplit:
        manifest_path = manifest_root / f"{dataset}_manifest.json"
        if not manifest_path.exists():
            print(f"❌ Manifest not found: {manifest_path}")
            continue
        check_unsplit_manifest(dataset, manifest_path)

    # Nhóm 2: manifest tách theo split
    for dataset in datasets_split:
        print(f"\n📁 {dataset}")
        for split in ["train", "dev", "test"]:
            manifest_path = manifest_root / f"{dataset}_{split}_manifest.json"
            if manifest_path.exists():
                check_split_manifest(dataset, split, manifest_path)
            else:
                print(f"  ❌ Missing manifest for {split}")

if __name__ == "__main__":
    main()

-------------------------

(stt310) ubuntu@ip-10-0-15-60:~/work/stt_chunkformer/chunkformer_vpb/preprocess$ python _3_wav_assert.py 
🎧 Checking sample rate: 8kHz expectation

📁 fpt_fosd: 25915 samples
📁 infore: 14935 samples
📁 lsvsc: 56823 samples
📁 speech_massive: 5120 samples
📁 vais1000: 1000 samples
📁 vietmed: 2858 samples
📁 vivos: 12420 samples
📁 vlsp2020: 56172 samples

📁 vi_voice
  ✅ train: 710217 samples checked
     👍 All files OK (8kHz)


  ✅ dev  : 88776 samples checked
     👍 All files OK (8kHz)
  ✅ test : 88779 samples checked
     👍 All files OK (8kHz)

📁 viet_bud500
  ✅ train: 634158 samples checked
     👍 All files OK (8kHz)
  ✅ dev  : 7500 samples checked
     👍 All files OK (8kHz)
  ✅ test : 7500 samples checked
     👍 All files OK (8kHz)

📁 vietspeech
  ✅ train: 820837 samples checked
     👍 All files OK (8kHz)
  ✅ dev  : 102604 samples checked
     👍 All files OK (8kHz)
  ✅ test : 102606 samples checked
     👍 All files OK (8kHz)

-------------------------

- Dữ liệu của mình đang là 8k - và bạn đã giúp mình check ok hết rồi nhé 
- Tập dữ liệu cơ bản có 2 loại như trong code: loại dùng field split (train/dev/test) và loại thì split luôn file manifest ra (loại dataset to train/dev/test) 

