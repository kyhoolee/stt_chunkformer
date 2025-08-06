import os
import json
import argparse
import time
from pathlib import Path
from collections import defaultdict
from multiprocessing import Pool
import gc

import torchaudio
from torchaudio.functional import resample
from chunkformer_vpb.training.data_augment import AudioAugmenter
from tqdm import tqdm

import torch
torch.set_num_threads(1)

AUG_TYPES = ["speed", "vol", "telephony", "noise"]
EXPECTED_ORIG_SR = 8000
TARGET_SR = 16000

BASE_INPUT_DIR = Path("~/dataset/data/data/processed/8khz").expanduser()
MANIFEST_INPUT_DIR = Path("~/dataset/data/manifests").expanduser()


# Global augmenter per-process
global_augmenter = None

def init_worker():
    """Initialize a single AudioAugmenter per process."""
    global global_augmenter
    global_augmenter = AudioAugmenter(sample_rate=TARGET_SR)


def process_one_sample(args):
    import time
    global global_augmenter  # Access per-process instance

    entry, dataset_name, split, output_root_str = args
    output_root = Path(output_root_str)

    try:
        utt_id = entry.get("utt_id", entry.get("key", "unknown_id"))
        key = entry.get("key", utt_id)
        wav_path = entry["wav"]
        input_audio_path = BASE_INPUT_DIR / dataset_name / split / "audio" / os.path.basename(wav_path)

        if not input_audio_path.exists():
            print(f"❌ [MISSING] {input_audio_path}")
            return {"error": f"Missing: {input_audio_path}"}

        print(f"\n🔍 Processing: {input_audio_path}")
        t0 = time.time()

        try:
            wav, sr = torchaudio.load(input_audio_path)
        except Exception as e:
            print(f"❌ [LOAD ERROR] {input_audio_path} - {e}")
            return {"error": f"Load failed: {input_audio_path}"}

        if sr != EXPECTED_ORIG_SR:
            print(f"⚠️  Invalid SR: {sr} (expected {EXPECTED_ORIG_SR})")
            return {"error": f"Invalid SR {sr}: {input_audio_path}"}

        try:
            wav_16k = resample(wav, orig_freq=sr, new_freq=TARGET_SR)
        except Exception as e:
            print(f"❌ [RESAMPLE ERROR] {input_audio_path} - {e}")
            return {"error": f"Resample failed: {input_audio_path}"}

        duration_sec = round(wav_16k.shape[1] / TARGET_SR, 3)

        audio_base_dir = output_root / "audio" / dataset_name / split
        base_name = os.path.basename(wav_path).replace(".wav", "_16k.wav")
        origin_path = audio_base_dir / "origin" / base_name
        origin_path.parent.mkdir(parents=True, exist_ok=True)
        torchaudio.save(origin_path, wav_16k, TARGET_SR)

        manifest_entries = [{
            "key": key,
            "utt_id": utt_id,
            "wav": f"origin/{base_name}",
            "text": entry.get("txt", ""),
            "split": split,
            "dataset": dataset_name,
            "duration": entry.get("duration", duration_sec),
            "original_wav": str(input_audio_path)
        }]

        aug_counter = defaultdict(int)
        for aug_type in AUG_TYPES:
            try:
                aug_wav = global_augmenter.apply(wav_16k.clone(), aug_type)
                aug_name = base_name.replace("_16k.wav", f"_{aug_type}.wav")
                aug_path = audio_base_dir / aug_type / aug_name
                aug_path.parent.mkdir(parents=True, exist_ok=True)
                torchaudio.save(aug_path, aug_wav, TARGET_SR)
                manifest_entries.append({
                    "key": key + f"__aug_{aug_type}",
                    "utt_id": utt_id + f"__aug_{aug_type}",
                    "wav": f"{aug_type}/{aug_name}",
                    "text": entry.get("txt", ""),
                    "split": split,
                    "dataset": dataset_name,
                    "duration": round(aug_wav.shape[1] / TARGET_SR, 3),
                    "original_wav": str(input_audio_path)
                })
                aug_counter[aug_type] += 1
            except Exception as e:
                print(f"❌ [AUG ERROR] {aug_type} - {e}")
                continue

        return {
            "manifest": manifest_entries,
            "origin": 1,
            "aug": dict(aug_counter)
        }

    except Exception as e:
        return {"error": f"{entry.get('wav', '???')}: {str(e)}"}


def process_dataset_split(dataset_name, split, mode, limit, log_every, num_workers):
    is_debug = mode == "debug"
    print(f"\n🟢 [{'DEBUG' if is_debug else 'FULL'}] {dataset_name} [{split}]")

    manifest_path = MANIFEST_INPUT_DIR / f"{dataset_name}_{split}_manifest.json"
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

    entries = [json.loads(line) for line in lines if line.strip()]
    args_list = [(entry, dataset_name, split, str(output_root)) for entry in entries]

    print(f"🚀 Using {num_workers} cores for {len(entries)} entries...")
    with Pool(processes=num_workers, initializer=init_worker) as pool:
        if is_debug:
            results = list(tqdm(pool.imap_unordered(process_one_sample, args_list), total=len(args_list)))
        else:
            results = []
            processed = 0
            start_time = time.time()
            for item in pool.imap_unordered(process_one_sample, args_list):
                results.append(item)
                processed += 1
                if processed % log_every == 0 or processed == len(args_list):
                    elapsed = time.time() - start_time
                    rate = processed / elapsed if elapsed > 0 else 0
                    eta = (len(args_list) - processed) / rate if rate > 0 else 0
                    percent = 100.0 * processed / len(args_list)
                    print(f"[{int(elapsed)}s] ⏱️ {processed}/{len(args_list)} ({percent:.1f}%) - ETA {int(eta)}s", flush=True)

    output_manifest = []
    aug_counter = defaultdict(int)
    origin_counter = 0
    errors = []

    for res in results:
        if "error" in res:
            errors.append(res["error"])
            continue
        output_manifest.extend(res["manifest"])
        origin_counter += res.get("origin", 0)
        for aug_type, count in res.get("aug", {}).items():
            aug_counter[aug_type] += count

    out_manifest_path = output_manifest_dir / f"{dataset_name}_{split}_manifest.json"
    with open(out_manifest_path, "w", encoding="utf-8") as f:
        for item in output_manifest:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"   📄 Manifest saved: {out_manifest_path}")
    print(f"   🎧 Origin files  : {origin_counter}")
    for aug_type in AUG_TYPES:
        print(f"   🎛️  {aug_type:<10}: {aug_counter[aug_type]}")
    print(f"   📊 Total entries : {len(output_manifest)}")

    if errors:
        print(f"\n⚠️  {len(errors)} errors encountered:")
        for err in errors[:10]:
            print("   ❌", err)
        if len(errors) > 10:
            print("   ...")


def run_benchmark_estimate(datasets, core_list, debug_n=100, output_path="benchmark_large.tsv"):
    results = []
    for ds in datasets:
        total = 0
        for split in ["train", "dev", "test"]:
            manifest_path = MANIFEST_INPUT_DIR / f"{ds}_{split}_manifest.json"
            if not manifest_path.exists():
                continue
            with open(manifest_path, "r", encoding="utf-8") as f:
                total += len(f.readlines())

        print(f"\n📦 Dataset: {ds} (Total: {total} samples)")

        for core in core_list:
            print(f"\n⚙️  Benchmarking {debug_n} samples per split with {core} cores...")
            start = time.time()
            for split in ["train", "dev", "test"]:
                process_dataset_split(
                    dataset_name=ds,
                    split=split,
                    mode="debug",
                    limit=debug_n,
                    log_every=debug_n + 1,
                    num_workers=core
                )
            end = time.time()
            time_real = round(end - start, 2)
            est_time = round((time_real / (debug_n * 3)) * total, 2)

            print(f"⏱️  Real time: {time_real}s → Estimated full time: {est_time}s")

            results.append({
                "dataset": ds,
                "cores": core,
                "sample_time": time_real,
                "total_samples": total,
                "est_full_time": est_time
            })

    with open(output_path, "w") as f:
        f.write("dataset\tcores\tsample_time_s\ttotal_samples\testimated_full_time_s\n")
        for row in results:
            f.write(f"{row['dataset']}\t{row['cores']}\t{row['sample_time']}\t{row['total_samples']}\t{row['est_full_time']}\n")
    print(f"\n📄 Benchmark results saved to: {output_path}")

    # gc.collect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", help="Dataset name (e.g. vi_voice)")
    parser.add_argument("--mode", choices=["debug", "full", "benchmark"], default="debug")
    parser.add_argument("--n", type=int, default=50, help="Debug sample limit")
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--num_workers", type=int, default=8)
    args = parser.parse_args()

    datasets_split = ["vi_voice", "viet_bud500", "vietspeech"]

    if args.mode == "benchmark":
        run_benchmark_estimate(datasets=datasets_split, core_list=[8, 1], debug_n=args.n)
    elif args.mode in ["debug", "full"]:
        for split in ["train", "dev", "test"]:
            process_dataset_split(
                dataset_name=args.dataset,
                split=split,
                mode=args.mode,
                limit=args.n,
                log_every=args.log_every,
                num_workers=args.num_workers
            )

