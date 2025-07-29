import json
import re
import os
from pathlib import Path
from tqdm import tqdm

AUG_TYPES = ['vol', 'speed', 'telephony', 'noise', 'pitch', 'reverb']
BASE_DIR = Path("../../../vpb_dataset").resolve()
WAVS_BASE = BASE_DIR / "archive/tts_dataset_best_call_agent_audio/tts_dataset_best_call_agent_audio"
MANIFEST_PATH = BASE_DIR / "manifest_vpb_non_empty_full/train_meta.json"
OUTPUT_MANIFEST_PATH = BASE_DIR / "manifest_vpb_non_empty_full/train_meta_aug.json"

def load_original_manifest():
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def clean_and_rename_augmented_audio_files(aug_type: str, original_utt_ids: set):
    aug_dir = WAVS_BASE / f"wavs_{aug_type}"
    if not aug_dir.exists():
        print(f"⚠️ Folder not found: {aug_dir}")
        return []

    all_files = list(aug_dir.glob("*.wav"))
    print(f"\n📁 {aug_type.upper():<10} | Total files: {len(all_files)}")

    valid = 0
    skipped = 0
    renamed = 0
    augmented_entries = []

    for wav_path in tqdm(all_files, desc=f"🔍 {aug_type}"):
        match = re.match(r"(.+?)_aug\d+\.wav", wav_path.name)
        base_utt_id = match.group(1) if match else wav_path.stem

        if base_utt_id not in original_utt_ids:
            skipped += 1
            continue

        valid += 1
        new_utt_id = f"{base_utt_id}_{aug_type}"
        new_filename = f"{new_utt_id}.wav"
        new_wav_path = wav_path.parent / new_filename

        if wav_path.name != new_filename:
            os.rename(wav_path, new_wav_path)
            renamed += 1
            print(f"🔁 Renamed: {wav_path.name} -> {new_filename}")

        rel_audio_path = new_wav_path.relative_to(BASE_DIR).as_posix()
        augmented_entries.append({
            "utt_id": new_utt_id,
            "audio_path": rel_audio_path,
            "augment_type": aug_type,
        })

    print(f"✅ Valid: {valid} | ❌ Skipped: {skipped} | 🔁 Renamed: {renamed}")
    return augmented_entries

def merge_with_text(augmented_entries, original_entries):
    uttid2text = {e["utt_id"]: e["text"] for e in original_entries}
    missing_text = 0

    for entry in augmented_entries:
        base_utt_id = entry["utt_id"].rsplit("_", 1)[0]
        entry["text"] = uttid2text.get(base_utt_id, "")
        if not entry["text"]:
            missing_text += 1

    if missing_text > 0:
        print(f"⚠️ Missing text for {missing_text} augmented entries")

    return augmented_entries

def main():
    print("📥 Loading original manifest...")
    original_entries = load_original_manifest()
    original_utt_ids = set(e["utt_id"] for e in original_entries)
    print(f"📊 Original entries: {len(original_entries)}")

    all_aug_entries = []
    for aug_type in AUG_TYPES:
        entries = clean_and_rename_augmented_audio_files(aug_type, original_utt_ids)
        all_aug_entries.extend(entries)

    print(f"\n📦 Total augmented entries (pre-merge): {len(all_aug_entries)}")

    # Merge với text
    all_aug_entries = merge_with_text(all_aug_entries, original_entries)

    # Deduplicate
    full_entries = {e["utt_id"]: e for e in original_entries + all_aug_entries}
    final_entries = list(full_entries.values())

    print(f"\n📑 FINAL SUMMARY")
    print(f"🟢 Original:       {len(original_entries)}")
    print(f"🟡 Augmented used: {len(all_aug_entries)}")
    print(f"🧮 Dedup total:    {len(final_entries)}")

    with open(OUTPUT_MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(final_entries, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Manifest saved: {OUTPUT_MANIFEST_PATH}")

if __name__ == "__main__":
    main()
