import os
import json
import argparse
from pathlib import Path
import torchaudio
from torchaudio.functional import resample
from chunkformer_vpb.training.data_augment import AudioAugmenter
from tqdm import tqdm
from collections import defaultdict

AUG_TYPES = ["speed", "vol", "telephony", "noise"]
EXPECTED_ORIG_SR = 8000
TARGET_SR = 16000

BASE_INPUT_DIR = Path("~/dataset/data/data/processed/8khz").expanduser()
MANIFEST_INPUT_DIR = Path("~/dataset/data/manifests").expanduser()

def process_dataset(dataset_name: str, mode: str = "debug", limit: int = 100):
    is_debug = mode == "debug"
    print(f"\n🟢 [{'DEBUG' if is_debug else 'FULL'}] Processing dataset: {dataset_name}")

    manifest_path = MANIFEST_INPUT_DIR / f"{dataset_name}_manifest.json"
    output_root = Path("~/stt/preprocess_debug" if is_debug else "~/stt/preprocess").expanduser()
    output_manifest_dir = output_root / "manifest" / dataset_name
    output_manifest_dir.mkdir(parents=True, exist_ok=True)

    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            if is_debug:
                lines = lines[:limit]
    except FileNotFoundError:
        print(f"❌ Manifest not found: {manifest_path}")
        return

    augmenter = AudioAugmenter(sample_rate=TARGET_SR)
    output_manifest = []
    aug_counter = defaultdict(int)
    origin_counter = 0

    for line in tqdm(lines, desc=f"🚀 {dataset_name}", unit="sample"):
        try:
            entry = json.loads(line)
            if "key" not in entry and "utt_id" not in entry:
                print("⚠️  Skipped entry: missing both 'key' and 'utt_id'")
                continue

            utt_id = entry.get("utt_id", entry.get("key", "unknown_id"))
            key = entry.get("key", utt_id)
            wav_path = entry["wav"]
            split = entry.get("split", "train")
            dataset = entry.get("dataset", dataset_name)

            input_audio_path = BASE_INPUT_DIR / dataset / split / "audio" / os.path.basename(wav_path)
            if not input_audio_path.exists():
                print(f"⚠️  Missing: {input_audio_path}")
                continue

            wav, sr = torchaudio.load(input_audio_path)
            if sr != EXPECTED_ORIG_SR:
                print(f"⚠️  Invalid SR {sr}: {input_audio_path}")
                continue

            wav_16k = resample(wav, orig_freq=sr, new_freq=TARGET_SR)
            duration_sec = round(wav_16k.shape[1] / TARGET_SR, 3)

            # Save origin
            audio_base_dir = output_root / "audio" / dataset / split
            origin_dir = audio_base_dir / "origin"
            origin_dir.mkdir(parents=True, exist_ok=True)
            base_name = os.path.basename(wav_path).replace(".wav", "_16k.wav")
            origin_path = origin_dir / base_name
            torchaudio.save(origin_path, wav_16k, TARGET_SR)

            output_manifest.append({
                "key": key,
                "utt_id": utt_id,
                "wav": f"origin/{base_name}",
                "text": entry.get("txt", ""),
                "split": split,
                "dataset": dataset,
                "duration": entry.get("duration", duration_sec),
                "original_wav": entry.get("original_wav", str(input_audio_path))
            })
            origin_counter += 1

            # Augment
            for aug_type in AUG_TYPES:
                aug_dir = audio_base_dir / aug_type
                aug_dir.mkdir(parents=True, exist_ok=True)

                aug_wav = augmenter.apply(wav_16k.clone(), aug_type)
                aug_name = base_name.replace("_16k.wav", f"_{aug_type}.wav")
                aug_path = aug_dir / aug_name
                torchaudio.save(aug_path, aug_wav, TARGET_SR)

                output_manifest.append({
                    "key": key + f"__aug_{aug_type}",
                    "utt_id": utt_id + f"__aug_{aug_type}",
                    "wav": f"{aug_type}/{aug_name}",
                    "text": entry.get("txt", ""),
                    "split": split,
                    "dataset": dataset,
                    "duration": round(aug_wav.shape[1] / TARGET_SR, 3),
                    "original_wav": entry.get("original_wav", str(input_audio_path))
                })
                aug_counter[aug_type] += 1

        except Exception as e:
            print(f"❌ Error on line: {e}")
            continue

    # Save manifest
    out_manifest_path = output_manifest_dir / f"{dataset_name}_manifest.json"
    with open(out_manifest_path, "w", encoding="utf-8") as f:
        for item in output_manifest:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"\n✅ [{mode.upper()} DONE] {dataset_name}")
    print(f"   📄 Manifest saved: {out_manifest_path}")
    print(f"   🎧 Origin files  : {origin_counter}")
    for aug_type in AUG_TYPES:
        print(f"   🎛️  {aug_type:<10}: {aug_counter[aug_type]}")
    print(f"   📊 Total entries : {len(output_manifest)}")

# ==== CLI ====
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="Dataset name (e.g. vietmed)")
    parser.add_argument("--mode", choices=["debug", "full"], default="debug", help="Run mode")
    parser.add_argument("--n", type=int, default=100, help="Number of samples in debug mode")
    args = parser.parse_args()

    if args.mode == 'debug':
        process_dataset(dataset_name=args.dataset, mode=args.mode, limit=args.n)
    else:
        datasets_unsplit = [
            "fpt_fosd", "infore", "lsvsc", "speech_massive", "vais1000",
            "vietmed", "vivos", "vlsp2020"
        ]

        for ds in datasets_unsplit:
            process_dataset(ds, mode="full")


# python -m chunkformer_vpb.preprocess._5_1_small_ds_parallel --dataset vietmed --mode debug --n 50
# python -m chunkformer_vpb.preprocess._5_1_small_ds_parallel --dataset vietmed --mode debug --n 50

# python -m chunkformer_vpb.preprocess._5_1_small_ds_parallel --mode full
