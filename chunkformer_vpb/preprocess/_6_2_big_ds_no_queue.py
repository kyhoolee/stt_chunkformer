#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import warnings
warnings.filterwarnings("ignore")

import os
import json
import argparse
import time
from pathlib import Path
from collections import defaultdict
from multiprocessing import Pool, cpu_count
import shutil

import torchaudio
from torchaudio.functional import resample

import torch
torch.set_num_threads(1)

from chunkformer_vpb.training.data_augment import AudioAugmenter

# ===== Config =====
AUG_TYPES = ["vol", "telephony", "noise"]  # bỏ speed
EXPECTED_ORIG_SR = 8000
TARGET_SR = 16000

BASE_INPUT_DIR = Path("~/dataset/data/data/processed/8khz").expanduser()
MANIFEST_INPUT_DIR = Path("~/dataset/data/manifests").expanduser()

# ===== Utils =====
def root_log(msg: str):
    print(msg, flush=True)

def read_jsonl_lines(p: Path, limit: int | None = None):
    with open(p, "r", encoding="utf-8") as f:
        if limit is None:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)
        else:
            c = 0
            for line in f:
                if c >= limit:
                    break
                line = line.strip()
                if line:
                    yield json.loads(line)
                    c += 1

def split_even(items, k):
    """Chia items thành k mảnh rời nhau (disjoint), cân bằng."""
    n = len(items)
    if k <= 1 or n == 0:
        return [items]
    size = (n + k - 1) // k
    return [items[i:i+size] for i in range(0, n, size)]

def clean_output_for_split(output_root: Path, dataset: str, split: str):
    """Xoá sạch output của split: audio + manifest file/folder."""
    audio_dir = output_root / "audio" / dataset / split
    man_dir   = output_root / "manifest" / dataset / split
    if audio_dir.exists():
        shutil.rmtree(audio_dir)
    audio_dir.mkdir(parents=True, exist_ok=True)
    if man_dir.exists():
        shutil.rmtree(man_dir)
    man_dir.mkdir(parents=True, exist_ok=True)

# ===== Per-process state =====
_global_augmenter = None

def init_worker():
    global _global_augmenter
    _global_augmenter = AudioAugmenter(sample_rate=TARGET_SR)

# ===== Worker: process a batch (disjoint chunk) with periodic progress logs =====
def process_chunk(entries, dataset_name, split, output_root_str, clip_guard: bool, log_every: int):
    """
    Mỗi worker xử lý một chunk rời nhau, ghi trực tiếp WAV ra đĩa.
    In log TIẾN ĐỘ theo chu kỳ log_every cho chunk này (không in per-entry).
    """
    global _global_augmenter
    pid = os.getpid()
    output_root = Path(output_root_str)

    audio_base_dir = output_root / "audio" / dataset_name / split
    (audio_base_dir / "origin").mkdir(parents=True, exist_ok=True)
    for a in AUG_TYPES:
        (audio_base_dir / a).mkdir(parents=True, exist_ok=True)

    manifest_lines = []
    origin_count = 0
    aug_counter = defaultdict(int)
    errors = []

    total = len(entries)
    start_time = time.time()
    last_report_t = start_time

    # header log cho worker này
    print(f"[PID {pid}][{split}] ▶️  start chunk: {total} items", flush=True)

    for idx, entry in enumerate(entries, 1):
        try:
            utt_id = entry.get("utt_id", entry.get("key", "unknown_id"))
            key = entry.get("key", utt_id)
            wav_field = entry["wav"]  # bắt buộc
            input_audio_path = BASE_INPUT_DIR / dataset_name / split / "audio" / os.path.basename(wav_field)

            if not input_audio_path.exists():
                errors.append(f"Missing: {input_audio_path}")
                continue

            # load 8k
            try:
                wav, sr = torchaudio.load(input_audio_path)
            except Exception as e:
                errors.append(f"Load failed: {input_audio_path} - {e}")
                continue

            if sr != EXPECTED_ORIG_SR:
                errors.append(f"Invalid SR {sr} (expect {EXPECTED_ORIG_SR}): {input_audio_path}")
                continue

            # resample -> 16k
            try:
                wav_16k = resample(wav, orig_freq=sr, new_freq=TARGET_SR)
            except Exception as e:
                errors.append(f"Resample failed: {input_audio_path} - {e}")
                continue

            duration_sec = round(wav_16k.shape[1] / TARGET_SR, 3)

            # save origin
            base_name = os.path.basename(wav_field).replace(".wav", "_16k.wav")
            origin_path = audio_base_dir / "origin" / base_name
            torchaudio.save(origin_path, wav_16k, TARGET_SR, format="WAV")
            origin_count += 1

            manifest_lines.append(json.dumps({
                "key": key,
                "utt_id": utt_id,
                "wav": f"origin/{base_name}",
                "text": entry.get("txt", ""),
                "split": split,
                "dataset": dataset_name,
                "duration": entry.get("duration", duration_sec),
                "original_wav": str(input_audio_path)
            }, ensure_ascii=False))

            # augment
            for aug_type in AUG_TYPES:
                try:
                    aug_wav = _global_augmenter.apply(wav_16k.clone(), aug_type)
                    if clip_guard:
                        peak = float(aug_wav.abs().max())
                        if peak >= 1.0:
                            aug_wav = (aug_wav / (peak + 1e-9)) * 0.999
                    aug_name = base_name.replace("_16k.wav", f"_{aug_type}.wav")
                    aug_path = audio_base_dir / aug_type / aug_name
                    torchaudio.save(aug_path, aug_wav, TARGET_SR, format="WAV")

                    manifest_lines.append(json.dumps({
                        "key": key + f"__aug_{aug_type}",
                        "utt_id": utt_id + f"__aug_{aug_type}",
                        "wav": f"{aug_type}/{aug_name}",
                        "text": entry.get("txt", ""),
                        "split": split,
                        "dataset": dataset_name,
                        "duration": round(aug_wav.shape[1] / TARGET_SR, 3),
                        "original_wav": str(input_audio_path)
                    }, ensure_ascii=False))
                    aug_counter[aug_type] += 1
                except Exception as e:
                    errors.append(f"AUG {aug_type} failed: {input_audio_path} - {e}")
                    continue

        except Exception as e:
            errors.append(f"{entry.get('wav', '???')}: {str(e)}")
            continue

        # ===== tiến độ theo chu kỳ =====
        if (idx % log_every == 0) or (idx == total):
            now = time.time()
            elapsed = now - start_time
            rate = idx / elapsed if elapsed > 0 else 0.0
            eta = (total - idx) / rate if rate > 0 else 0.0
            print(f"[PID {pid}][{split}] ⏱️ {idx}/{total} ({100.0*idx/total:.1f}%) "
                  f"- {rate:.2f} it/s - ETA {int(eta)}s", flush=True)
            last_report_t = now

    print(f"[PID {pid}][{split}] ✅ done chunk: {total} items", flush=True)
    return manifest_lines, origin_count, dict(aug_counter), errors

# ===== Split-level processing =====
def process_dataset_split(dataset_name: str,
                          split: str,
                          mode: str,
                          limit: int,
                          num_workers: int,
                          clip_guard: bool,
                          log_every: int):
    is_debug = (mode == "debug")
    root_log(f"\n🟢 [{'DEBUG' if is_debug else 'FULL'}] {dataset_name} [{split}]")

    # input manifest
    manifest_path = MANIFEST_INPUT_DIR / f"{dataset_name}_{split}_manifest.json"
    if not manifest_path.exists():
        root_log(f"⚠️  Skip {dataset_name}/{split}: manifest not found: {manifest_path}")
        return

    # output root
    output_root = Path("~/stt/preprocess_debug" if is_debug else "~/stt/preprocess").expanduser()
    # clean split outputs fully (simple/stateless)
    clean_output_for_split(output_root, dataset_name, split)

    # read entries
    lines_limit = (limit if is_debug else None)
    entries = list(read_jsonl_lines(manifest_path, limit=lines_limit))
    n = len(entries)
    if n == 0:
        root_log("⚠️  Empty manifest, skip.")
        return

    # split disjoint chunks
    k = max(1, num_workers)
    chunks = split_even(entries, k)
    root_log(f"🚀 Using {k}/{cpu_count()} workers for {n} entries (chunks={len(chunks)})")

    start = time.time()
    origin_total = 0
    aug_total = defaultdict(int)
    errors_all = []
    manifest_lines_all = []

    # run workers
    with Pool(processes=k, initializer=init_worker) as pool:
        jobs = []
        for ch in chunks:
            jobs.append(pool.apply_async(
                process_chunk,
                (ch, dataset_name, split, str(output_root), clip_guard, log_every)
            ))

        for j in jobs:
            manifest_lines, origin_count, aug_counter, errors = j.get()
            manifest_lines_all.extend(manifest_lines)
            origin_total += origin_count
            for k2, v2 in aug_counter.items():
                aug_total[k2] += v2
            errors_all.extend(errors)

    # write manifest (single file per split)
    out_manifest_dir = output_root / "manifest" / dataset_name / split
    out_manifest_dir.mkdir(parents=True, exist_ok=True)
    out_manifest_path = out_manifest_dir / f"{dataset_name}_{split}_manifest.json"
    with open(out_manifest_path, "w", encoding="utf-8") as f:
        for line in manifest_lines_all:
            f.write(line + "\n")

    # summary
    elapsed = time.time() - start
    root_log(f"   📄 Manifest saved: {out_manifest_path}")
    root_log(f"   🎧 Origin files  : {origin_total}")
    for a in AUG_TYPES:
        root_log(f"   🎛️  {a:<10}: {aug_total[a]}")
    root_log(f"   📊 Total entries : {len(manifest_lines_all)}")
    root_log(f"   ⏱️  Elapsed      : {elapsed:.1f}s")

    if errors_all:
        root_log(f"\n⚠️  {len(errors_all)} errors (showing up to 20):")
        for e in errors_all[:20]:
            root_log(f"   ❌ {e}")
        if len(errors_all) > 20:
            root_log("   ...")

# ===== Dataset-level orchestrator =====
def process_dataset_big(dataset_name: str,
                        mode: str = "debug",
                        limit: int = 100,
                        num_workers: int = 8,
                        clip_guard: bool = True,
                        log_every: int = 2000):
    for split in ["train", "dev", "test"]:
        process_dataset_split(
            dataset_name=dataset_name,
            split=split,
            mode=mode,
            limit=limit,
            num_workers=num_workers,
            clip_guard=clip_guard,
            log_every=log_every,
        )

# ===== CLI =====
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True,
                   help="Big dataset name (vi_voice, viet_bud500, vietspeech)")
    p.add_argument("--mode", choices=["debug", "full"], default="debug")
    p.add_argument("--n", type=int, default=100, help="Samples per split in debug mode")
    p.add_argument("--num_workers", type=int, default=16)
    p.add_argument("--no_clip_guard", action="store_true",
                   help="Disable soft-clip guard after vol augment")
    p.add_argument("--log_every", type=int, default=2000,
                   help="Per-process progress log interval (number of items).")
    args = p.parse_args()

    root_log(f"ARGS: dataset={args.dataset} mode={args.mode} n={args.n} "
             f"num_workers={args.num_workers} clip_guard={not args.no_clip_guard} "
             f"log_every={args.log_every}")

    process_dataset_big(
        dataset_name=args.dataset,
        mode=args.mode,
        limit=args.n,
        num_workers=args.num_workers,
        clip_guard=not args.no_clip_guard,
        log_every=args.log_every,
    )

if __name__ == "__main__":
    main()
