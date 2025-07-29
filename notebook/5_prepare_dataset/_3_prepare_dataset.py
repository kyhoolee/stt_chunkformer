import json
import torchaudio
import torchaudio.functional as F
from pathlib import Path
from tqdm import tqdm
import argparse
import time
import multiprocessing as mp
from functools import partial

from chunkformer_vpb.training.data_augment import AudioAugmenter

AUG_TYPES = [
                'vol', 'speed', 'telephony', 
                'noise', 'pitch', 'reverb'
             ]
SAMPLE_RATE = 16000
N_SAMPLE_TEST = 160

def apply_augment(wav, augmenter, aug_type: str):
    return augmenter.apply(wav, aug_type)

def process_one_entry(entry, base_dir: Path, test_mode=False):
    try:
        audio_path = Path(entry["audio_path"])
        wav_path = base_dir / audio_path

        if not wav_path.exists():
            return f"⚠️ Missing: {wav_path}", []

        # Load waveform
        wav, sr = torchaudio.load(wav_path)
        if sr != SAMPLE_RATE:
            wav = F.resample(wav, orig_freq=sr, new_freq=SAMPLE_RATE)

        duration_sec = wav.shape[1] / SAMPLE_RATE
        result_logs = []
        new_entries = []

        if test_mode:
            result_logs.append(f"\n🎧 Processing: {entry['utt_id']}")
            result_logs.append(f"   🔹 Input wav: {wav_path}")
            result_logs.append(f"   📏 Duration: {duration_sec:.2f} sec")

        augmenter = AudioAugmenter(sample_rate=SAMPLE_RATE)

        for i, aug_type in enumerate(AUG_TYPES):
            new_utt_id = entry["utt_id"] + f"_aug{i+1}"
            new_wav_folder = audio_path.parent.name.replace("wavs", f"wavs_{aug_type}")
            new_wav_rel_path = audio_path.parent.parent / new_wav_folder / (audio_path.stem + f"_aug{i+1}.wav")
            new_wav_abs_path = base_dir / new_wav_rel_path

            if new_wav_abs_path.exists():
                # if test_mode:
                #     result_logs.append(f"⏩ Skipped (exists): {new_wav_abs_path}")
                print(f"⏩ Skipped (exists): {new_wav_abs_path}", flush=True)
            else:
                new_wav_abs_path.parent.mkdir(parents=True, exist_ok=True)
                aug_wav = apply_augment(wav, augmenter, aug_type)
                torchaudio.save(new_wav_abs_path, aug_wav, SAMPLE_RATE)
                print(f"🔁 {aug_type:<10} | {new_utt_id:<20} | {duration_sec:.2f} sec", flush=True)

                if test_mode:
                    result_logs.append(f"✅ Saved ({aug_type}): {new_wav_abs_path}")

            # Luôn thêm vào manifest, kể cả khi wav đã tồn tại
            new_entry = {
                "utt_id": new_utt_id,
                "audio_path": str(new_wav_rel_path).replace("\\", "/"),
                "text": entry["text"],
                "augment_type": aug_type
            }
            new_entries.append(new_entry)


        return "\n".join(result_logs), new_entries

    except Exception as e:
        print(f"❌ ERROR with {entry['utt_id']}: {e}", flush=True)
        return f"❌ Error: {e}", []

def expand_and_generate_audio(base_dir: Path, manifest_path: Path, output_manifest_path: Path, test_mode=False, num_workers=4, serial_debug=False):
    with open(manifest_path, "r", encoding="utf-8") as f:
        entries = json.load(f)

    if test_mode:
        entries = entries[:N_SAMPLE_TEST]
        print(f"\n🧪 [TEST MODE] Chạy thử {len(entries)} samples...\n")

    start_time = time.time()

    if serial_debug:
        print("🐛 [DEBUG MODE] Chạy tuần tự từng sample...\n")
        results = [process_one_entry(entry, base_dir, test_mode) for entry in tqdm(entries)]
    else:
        if num_workers is None:
            num_workers = max(mp.cpu_count() - 2, 1)
        chunksize = 8

        print(f"🚀 Using {num_workers} workers with chunksize={chunksize}")

        with mp.Pool(processes=num_workers) as pool:
            results = list(tqdm(
                pool.imap_unordered(partial(process_one_entry, base_dir=base_dir, test_mode=test_mode), entries),
                total=len(entries),
                desc="🔁 Augmenting"
            ))

    all_logs = []
    all_augmented_entries = []

    # for log, new_entries in results:
    #     if test_mode and log:
    #         print(log)
    #     all_augmented_entries.extend(new_entries)

    full_entries = entries + all_augmented_entries
    with open(output_manifest_path, "w", encoding="utf-8") as f:
        json.dump(full_entries, f, ensure_ascii=False, indent=2)

    total_time = time.time() - start_time
    print(f"\n✅ Manifest saved: {output_manifest_path.name}")
    print(f"📦 Original: {len(entries)} | Augmented: {len(all_augmented_entries)} | Total: {len(full_entries)}")
    print(f"⏱️ Total time: {total_time:.2f} sec | Avg per sample: {total_time / len(entries):.2f} sec")

# === CLI ===
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true", help="Run in test mode (few samples only)")
    parser.add_argument("--num-workers", type=int, default=None, help="Number of parallel processes")
    parser.add_argument("--serial", action="store_true", help="Run serially for debugging")
    args = parser.parse_args()

    base_dir = Path("../../../vpb_dataset").resolve()
    manifest_path = base_dir / "manifest_vpb_non_empty_full" / "train_meta.json"
    output_manifest_path = base_dir / "manifest_vpb_non_empty_full" / "train_meta_augmented.json"

    expand_and_generate_audio(
        base_dir=base_dir,
        manifest_path=manifest_path,
        output_manifest_path=output_manifest_path,
        test_mode=args.test,
        num_workers=args.num_workers,
        serial_debug=args.serial
    )


# python -u _3_prepare_dataset.py --test --num-workers 16
# python -u _3_prepare_dataset.py --test --serial
# nohup python -u _3_prepare_dataset.py --num-workers 16 > log_aug.txt 2>&1 &


