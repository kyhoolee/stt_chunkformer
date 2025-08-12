import os
import json
import argparse
import time
from pathlib import Path
from collections import defaultdict
from multiprocessing import Pool, cpu_count
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

# Shared per-process augmenter
_GLOBAL_AUGMENTER = None



def init_worker():
    global _GLOBAL_AUGMENTER
    _GLOBAL_AUGMENTER = AudioAugmenter(sample_rate=TARGET_SR)

def process_one_sample(args):
    entry, dataset_name, output_root_str = args
    output_root = Path(output_root_str)
    global _GLOBAL_AUGMENTER

    try:
        utt_id = entry.get("utt_id", entry.get("key", "unknown_id"))
        key = entry.get("key", utt_id)
        wav_path = entry["wav"]
        split = entry.get("split", "train")
        dataset = entry.get("dataset", dataset_name)

        input_audio_path = BASE_INPUT_DIR / dataset / split / "audio" / os.path.basename(wav_path)
        if not input_audio_path.exists():
            return {"error": f"Missing: {input_audio_path}"}

        wav, sr = torchaudio.load(input_audio_path)
        if sr != EXPECTED_ORIG_SR:
            return {"error": f"Invalid SR {sr}: {input_audio_path}"}

        wav_16k = resample(wav, orig_freq=sr, new_freq=TARGET_SR)
        duration_sec = round(wav_16k.shape[1] / TARGET_SR, 3)

        # Output folders
        audio_base_dir = output_root / "audio" / dataset / split
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
            "dataset": dataset,
            "duration": entry.get("duration", duration_sec),
            "original_wav": str(input_audio_path)
        }]

        aug_counter = defaultdict(int)

        for aug_type in AUG_TYPES:
            aug_wav = _GLOBAL_AUGMENTER.apply(wav_16k.clone(), aug_type)
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
                "dataset": dataset,
                "duration": round(aug_wav.shape[1] / TARGET_SR, 3),
                "original_wav": str(input_audio_path)
            })
            aug_counter[aug_type] += 1

        return {
            "manifest": manifest_entries,
            "origin": 1,
            "aug": dict(aug_counter)
        }

    except Exception as e:
        return {"error": f"{entry.get('wav', '???')}: {str(e)}"}


def progress_logger(pool, iterable, total, log_every=1000):
    results = []
    processed = 0
    start_time = time.time()

    for item in pool.imap_unordered(process_one_sample, iterable):
        results.append(item)
        processed += 1

        if processed % log_every == 0 or processed == total:
            elapsed = time.time() - start_time
            rate = processed / elapsed if elapsed > 0 else 0
            eta = (total - processed) / rate if rate > 0 else 0
            percent = 100.0 * processed / total
            print(f"[{int(elapsed)}s] ⏱️ {processed}/{total} ({percent:.1f}%) - ETA {int(eta)}s", flush=True)

    return results


def process_dataset(dataset_name: str, mode: str = "debug", limit: int = 100, log_every: int = 50, num_workers: int = 4):
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

    entries = [json.loads(line) for line in lines if line.strip()]
    args_list = [(entry, dataset_name, str(output_root)) for entry in entries]

    print(f"🚀 Using {num_workers} cores for {len(entries)} entries...")

    with Pool(processes=num_workers, initializer=init_worker) as pool:
        if is_debug:
            results = list(tqdm(pool.imap_unordered(process_one_sample, args_list), total=len(args_list)))
        else:
            results = progress_logger(pool, args_list, total=len(args_list), log_every=log_every)

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

    if errors:
        print(f"\n⚠️  {len(errors)} errors encountered:")
        for err in errors[:10]:
            print("   ❌", err)
        if len(errors) > 10:
            print("   ...")



def run_benchmark_estimate(datasets, core_list, debug_n=100, output_path="benchmark_result.tsv"):
    results = []
    for ds in datasets:
        # Load để đếm tổng số mẫu thật
        manifest_path = MANIFEST_INPUT_DIR / f"{ds}_manifest.json"
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                total_lines = len(f.readlines())
        except FileNotFoundError:
            print(f"❌ Manifest not found for dataset: {ds}")
            continue

        print(f"\n📦 Dataset: {ds} (Total: {total_lines} samples)")

        for core in core_list:
            print(f"\n⚙️  Benchmarking {debug_n} samples with {core} cores...")
            start = time.time()
            process_dataset(
                dataset_name=ds,
                mode="debug",
                limit=debug_n,
                log_every=debug_n + 1,  # để không log gì
                num_workers=core
            )
            end = time.time()
            time_real = round(end - start, 2)
            est_time = round((time_real / debug_n) * total_lines, 2)

            print(f"⏱️  Real time: {time_real}s → Estimated full time: {est_time}s")

            results.append({
                "dataset": ds,
                "cores": core,
                "sample_time": time_real,
                "total_samples": total_lines,
                "est_full_time": est_time
            })

    # Save TSV
    with open(output_path, "w") as f:
        f.write("dataset\tcores\tsample_time_s\ttotal_samples\testimated_full_time_s\n")
        for row in results:
            f.write(f"{row['dataset']}\t{row['cores']}\t{row['sample_time']}\t{row['total_samples']}\t{row['est_full_time']}\n")
    print(f"\n📄 Benchmark results saved to: {output_path}")

    gc.collect()
    time.sleep(1)


import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=False, help="Single dataset name (e.g. vietmed)")
    parser.add_argument("--datasets", nargs="+", help="List of dataset names (e.g. vietmed vivos)")
    parser.add_argument("--mode", choices=["debug", "full", "benchmark"], default="debug", help="Run mode")
    parser.add_argument("--n", type=int, default=100, help="Number of samples in debug/benchmark mode")
    parser.add_argument("--log_every", type=int, default=50, help="Log every N samples (for full mode)")
    parser.add_argument("--num_workers", type=int, default=16, help="Number of processes for multiprocessing")
    args = parser.parse_args()

    DEFAULT_DATASETS = [
        "fpt_fosd", 
        "infore", 
        "lsvsc", 
        "speech_massive", 
        "vais1000",
        "vietmed", 
        "vivos", 
        "vlsp2020"
    ]

    def run(ds_name):
        process_dataset(
            dataset_name=ds_name,
            mode=args.mode,
            limit=args.n,
            log_every=args.log_every,
            num_workers=args.num_workers
        )

    selected_datasets = args.datasets or DEFAULT_DATASETS

    if args.mode == 'debug':
        run(args.dataset or selected_datasets[0])
    elif args.mode == 'full':
        for ds in selected_datasets:
            run(ds)
    elif args.mode == 'benchmark':
        core_list = [8, 4, 1]
        run_benchmark_estimate(
            datasets=selected_datasets,
            core_list=core_list,
            debug_n=args.n,
            output_path="benchmark_result.tsv"
        )
