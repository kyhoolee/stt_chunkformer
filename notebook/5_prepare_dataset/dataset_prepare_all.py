import os
import json
import random
from pathlib import Path

def collect_transcripts(label_dirs, audio_dir, output_dir, split_ratio=0.8):
    """
    label_dirs: list of folders chứa transcript đã được sửa
    audio_dir : folder chứa các file wav gốc
    output_dir: folder để lưu train_meta.json & valid_meta.json
    """
    audio_dir  = Path(audio_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_entries = []
    seen_utt = set()

    for label_dir in label_dirs:
        label_dir = Path(label_dir)
        print(f"🔍 Processing label_dir: {label_dir}")

        for txt_file in sorted(label_dir.glob("*.txt")):
            utt_id = txt_file.stem
            if utt_id.startswith(".") or utt_id in seen_utt:
                continue  # bỏ .progress hoặc trùng id
            wav_path = audio_dir / f"{utt_id}.wav"
            if not wav_path.exists():
                print(f"⚠️ Missing audio for: {utt_id}")
                continue
            with open(txt_file, "r", encoding="utf-8") as f:
                text = f.read().strip()
            if not text:
                continue
            entry = {
                "utt_id": utt_id,
                "audio_path": str(wav_path.relative_to(output_dir.parent)),
                "text": text
            }
            all_entries.append(entry)
            seen_utt.add(utt_id)

    print(f"\n✅ Tổng số sample hợp lệ sau khi gộp: {len(all_entries)}")

    # Shuffle + Split
    random.shuffle(all_entries)
    split_idx = int(len(all_entries) * split_ratio)
    train_entries = all_entries[:split_idx]
    valid_entries = all_entries[split_idx:]

    # Write
    with open(output_dir / "train_meta.json", "w", encoding="utf-8") as f:
        json.dump(train_entries, f, ensure_ascii=False, indent=2)
    with open(output_dir / "valid_meta.json", "w", encoding="utf-8") as f:
        json.dump(valid_entries, f, ensure_ascii=False, indent=2)

    print(f"📄 train_meta.json: {len(train_entries)} samples")
    print(f"📄 valid_meta.json: {len(valid_entries)} samples")


label_dirs = [
    "../label/transcripts_corrected_quangdm4_cutoff_21_7/transcripts_corrected",
    "../label/transcript_corrected_hieudm13_cutoff_20250721/transcript_corrected"
]

audio_dir  = "../archive/tts_dataset_best_call_agent_audio/tts_dataset_best_call_agent_audio/wavs"
output_dir = "../manifest_vpb_all"

collect_transcripts(label_dirs, audio_dir, output_dir)
