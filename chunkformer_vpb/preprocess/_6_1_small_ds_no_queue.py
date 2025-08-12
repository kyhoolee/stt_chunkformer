import os
import json
import time
from datetime import datetime
from pathlib import Path
from collections import defaultdict
from multiprocessing import Pool, cpu_count
import gc

import torchaudio
from torchaudio.functional import resample
from tqdm import tqdm
import torch
torch.set_num_threads(1)

from chunkformer_vpb.training.data_augment import AudioAugmenter

# ===== Config =====
AUG_TYPES = [
    # "speed", 
             "vol", "telephony", "noise"]
EXPECTED_ORIG_SR = 8000
TARGET_SR = 16000

BASE_INPUT_DIR = Path("~/dataset/data/data/processed/8khz").expanduser()
MANIFEST_INPUT_DIR = Path("~/dataset/data/manifests").expanduser()

# ===== Utils (logging + timers) =====
def _now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def log(msg, level="INFO"):
    print(f"[{_now()}][{level}] {msg}", flush=True)

class Stopwatch:
    """Simple timer with named laps."""
    def __init__(self):
        self.t0 = time.perf_counter()
        self.last = self.t0
        self.laps = {}

    def lap(self, name):
        t = time.perf_counter()
        self.laps[name] = self.laps.get(name, 0.0) + (t - self.last)
        self.last = t

    def total(self):
        return time.perf_counter() - self.t0

    def report(self):
        parts = [f"{k}={v:.3f}s" for k, v in self.laps.items()]
        return ", ".join(parts) + (", " if parts else "") + f"total={self.total():.3f}s"

# ===== Worker Init =====
_GLOBAL_AUGMENTER = None
_VERBOSE = 1  # 0: minimal, 1: normal, 2: very verbose

def init_worker(verbose=1):
    global _GLOBAL_AUGMENTER, _VERBOSE
    _GLOBAL_AUGMENTER = AudioAugmenter(sample_rate=TARGET_SR)
    _VERBOSE = verbose
    if _VERBOSE >= 2:
        log("Worker initialized with AudioAugmenter", level="DEBUG")

# ===== Process 1 sample =====
def process_one_sample(entry, dataset_name, output_root_str):
    output_root = Path(output_root_str)
    global _GLOBAL_AUGMENTER, _VERBOSE

    sw = Stopwatch()
    try:
        utt_id = entry.get("utt_id", entry.get("key", "unknown_id"))
        key = entry.get("key", utt_id)
        wav_path = entry["wav"]
        split = entry.get("split", "train")
        dataset = entry.get("dataset", dataset_name)

        # 1) Resolve input path
        input_audio_path = BASE_INPUT_DIR / dataset / split / "audio" / os.path.basename(wav_path)
        if not input_audio_path.exists():
            return {"error": f"Missing: {input_audio_path}"}
        if _VERBOSE >= 2:
            log(f"[sample {key}] Input: {input_audio_path}", level="DEBUG")

        # 2) Load
        wav, sr = torchaudio.load(input_audio_path)
        sw.lap("load_wav")
        if sr != EXPECTED_ORIG_SR:
            return {"error": f"Invalid SR {sr}: {input_audio_path}"}

        # 3) Resample
        wav_16k = resample(wav, orig_freq=sr, new_freq=TARGET_SR)
        sw.lap("resample")
        duration_sec = round(wav_16k.shape[1] / TARGET_SR, 3)

        # 4) Save origin
        audio_base_dir = output_root / "audio" / dataset / split
        base_name = os.path.basename(wav_path).replace(".wav", "_16k.wav")
        origin_path = audio_base_dir / "origin" / base_name
        origin_path.parent.mkdir(parents=True, exist_ok=True)
        torchaudio.save(origin_path, wav_16k, TARGET_SR)
        sw.lap("save_origin")

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

        # 5) Augment (timed per aug)
        aug_counter = defaultdict(int)
        for aug_type in AUG_TYPES:
            t_aug_start = time.perf_counter()
            aug_wav = _GLOBAL_AUGMENTER.apply(wav_16k.clone(), aug_type)
            t_aug_apply = time.perf_counter()

            aug_name = base_name.replace("_16k.wav", f"_{aug_type}.wav")
            aug_path = audio_base_dir / aug_type / aug_name
            aug_path.parent.mkdir(parents=True, exist_ok=True)
            torchaudio.save(aug_path, aug_wav, TARGET_SR)
            t_aug_save = time.perf_counter()

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

            # Record laps
            sw.laps[f"aug_apply[{aug_type}]"] = sw.laps.get(f"aug_apply[{aug_type}]", 0.0) + (t_aug_apply - t_aug_start)
            sw.laps[f"aug_save[{aug_type}]"] = sw.laps.get(f"aug_save[{aug_type}]", 0.0) + (t_aug_save - t_aug_apply)

            if _VERBOSE >= 2:
                log(f"[sample {key}] aug={aug_type} apply={t_aug_apply - t_aug_start:.3f}s save={t_aug_save - t_aug_apply:.3f}s", level="DEBUG")

        sw.lap("augment_total")

        if _VERBOSE >= 2:
            log(f"[sample {key}] {sw.report()}", level="DEBUG")

        return {
            "manifest": manifest_entries,
            "origin": 1,
            "aug": dict(aug_counter),
            "timing": sw.laps,
            "timing_total": sw.total(),
        }

    except Exception as e:
        return {"error": f"{entry.get('wav', '???')}: {str(e)}"}

# ===== Process a chunk =====
def process_chunk(index, chunk_entries, dataset_name, output_root_str, verbose=1):
    """Mỗi process chỉ xử lý chunk này."""
    global _VERBOSE
    _VERBOSE = verbose
    sw_chunk = Stopwatch()
    chunk_results = []

    if _VERBOSE >= 1:
        log(f"[chunk {index}] start: {len(chunk_entries)} entries", level="DEBUG")

    for i, entry in enumerate(chunk_entries, 1):
        res = process_one_sample(entry, dataset_name, output_root_str)
        chunk_results.append(res)
        # Print progress per 10 samples (or always if very verbose)
        if _VERBOSE >= 2 or (i % 10 == 0):
            key_show = entry.get('key', entry.get('utt_id', 'unknown_key'))
            log(f"✅ [chunk {index}] processed {i}/{len(chunk_entries)} (last={key_show})", level="DEBUG")

    sw_chunk.lap("process_entries")

    # quick timing summary for the chunk
    ok = sum(1 for r in chunk_results if "error" not in r)
    err = len(chunk_results) - ok
    log(f"[chunk {index}] done: ok={ok}, err={err}, {sw_chunk.report()}", level="DEBUG")
    return chunk_results

# ===== Main processing =====
def process_dataset(dataset_name: str, mode: str = "debug", limit: int = 100, num_workers: int = 4, verbose: int = 1):
    is_debug = mode == "debug"
    sw_ds = Stopwatch()

    log(f"🟢 [{'DEBUG' if is_debug else 'FULL'}] Processing dataset: {dataset_name}")
    log(f"   • workers={num_workers}/{cpu_count()} • mode={mode} • limit={limit} • verbose={verbose}")

    manifest_path = MANIFEST_INPUT_DIR / f"{dataset_name}_manifest.json"
    output_root = Path("~/stt/preprocess_debug" if is_debug else "~/stt/preprocess").expanduser()
    output_manifest_dir = output_root / "manifest" / dataset_name
    output_manifest_dir.mkdir(parents=True, exist_ok=True)

    # Load manifest lines
    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            if is_debug:
                lines = lines[:limit]
        sw_ds.lap("read_manifest")
        log(f"   • manifest_lines={len(lines)} from {manifest_path}")
    except FileNotFoundError:
        log(f"❌ Manifest not found: {manifest_path}", level="ERROR")
        return

    entries = [json.loads(line) for line in lines if line.strip()]
    sw_ds.lap("parse_manifest")

    # Split into chunks
    chunk_size = (len(entries) + num_workers - 1) // num_workers
    chunks = [entries[i:i + chunk_size] for i in range(0, len(entries), chunk_size)]
    log(f"🚀 Split into {len(chunks)} chunks (chunk_size≈{chunk_size})")
    sw_ds.lap("split_chunks")

    # Process with Pool
    log("🧵 Spawning workers...", level="DEBUG")
    with Pool(processes=num_workers, initializer=init_worker, initargs=(verbose,)) as pool:
        all_results = pool.starmap(
            process_chunk,
            [(index, chunk, dataset_name, str(output_root), verbose) for index, chunk in enumerate(chunks)]
        )
    sw_ds.lap("pool_process")

    # Aggregate results
    output_manifest = []
    aug_counter = defaultdict(int)
    origin_counter = 0
    errors = []
    timing_accum = defaultdict(float)
    sample_count = 0

    for chunk_results in all_results:
        for res in chunk_results:
            if "error" in res:
                errors.append(res["error"])
                continue
            output_manifest.extend(res["manifest"])
            origin_counter += res.get("origin", 0)
            for aug_type, count in res.get("aug", {}).items():
                aug_counter[aug_type] += count
            # timing
            for k, v in res.get("timing", {}).items():
                timing_accum[k] += v
            sample_count += 1

    sw_ds.lap("aggregate")

    # Save manifest
    out_manifest_path = output_manifest_dir / f"{dataset_name}_manifest.json"
    with open(out_manifest_path, "w", encoding="utf-8") as f:
        for item in output_manifest:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    sw_ds.lap("save_manifest")

    # Print summary
    log(f"\n✅ [{mode.upper()} DONE] {dataset_name}")
    log(f"   📄 Manifest saved: {out_manifest_path}")
    log(f"   🎧 Origin files  : {origin_counter}")
    for aug_type in AUG_TYPES:
        log(f"   🎛️  {aug_type:<10}: {aug_counter[aug_type]}")
    log(f"   📊 Total entries : {len(output_manifest)}")

    if sample_count > 0:
        # avg timings per sample where available
        avg = {k: timing_accum[k] / sample_count for k in timing_accum}
        log("⏱️  Avg per-sample timings:")
        for k in sorted(avg.keys()):
            log(f"     • {k:<20} {avg[k]:.4f}s")
    log(f"⏲️  Dataset timing: {sw_ds.report()}")

    if errors:
        log(f"\n⚠️  {len(errors)} errors encountered (showing up to 10):", level="WARN")
        for err in errors[:10]:
            log(f"   ❌ {err}", level="WARN")
        if len(errors) > 10:
            log("   ...", level="WARN")

    gc.collect()

# ===== CLI =====
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", help="Single dataset name (e.g. vietmed)")
    parser.add_argument("--datasets", nargs="+", help="List of dataset names")
    parser.add_argument("--mode", choices=["debug", "full"], default="debug")
    parser.add_argument("--n", type=int, default=100, help="Number of samples in debug mode")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of processes")
    parser.add_argument("--verbose", type=int, default=1, choices=[0,1,2], help="Log verbosity (0-2)")
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

    print(f"datset-args")
    print("DATASETS::", args.datasets)
    print("DATASET::", args.dataset)

    ds = args.datasets or [args.dataset] if args.dataset else args.datasets
    datasets_to_run = ds if ds else DEFAULT_DATASETS

    for ds in datasets_to_run:
        process_dataset(
            dataset_name=ds,
            mode=args.mode,
            limit=args.n,
            num_workers=args.num_workers,
            verbose=args.verbose
        )
