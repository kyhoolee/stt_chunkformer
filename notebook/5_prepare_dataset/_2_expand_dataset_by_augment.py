import json
import argparse
from pathlib import Path

# === Cấu hình ===
AUG_TYPES = ['vol', 'speed', 'telephony', 'noise', 'pitch', 'reverb']

def expand_manifest(manifest_path: Path, output_path: Path):
    with open(manifest_path, "r", encoding="utf-8") as f:
        entries = json.load(f)

    NUM_AUG = len(AUG_TYPES)

    augmented = []
    for entry in entries:
        for i in range(NUM_AUG):
            aug_type = AUG_TYPES[i % len(AUG_TYPES)]
            new_entry = {
                "utt_id": entry["utt_id"] + f"_aug{i+1}",
                "audio_path": entry["audio_path"],
                "text": entry["text"],
                "augment_type": aug_type
            }
            augmented.append(new_entry)

    full_entries = entries + augmented

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(full_entries, f, ensure_ascii=False, indent=2)

    print(f"✅ Saved: {output_path.name}")
    print(f"📦 Original: {len(entries)} | Augmented: {len(augmented)} | Total: {len(full_entries)}")


# === CLI ===
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=str, default="train_meta.json", help="Tên file manifest gốc")
    parser.add_argument("--output", type=str, default="train_meta_augmented.json", help="Tên file manifest đầu ra")
    args = parser.parse_args()

    # Base dir = ../../vpb_dataset
    base_dir = Path("../../../vpb_dataset").resolve()
    
    manifest_dir = base_dir / "manifest_vpb_non_empty_full"

    manifest_path = manifest_dir / args.manifest
    output_path = manifest_dir / args.output

    expand_manifest(manifest_path=manifest_path, output_path=output_path)
