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
import gc

import torchaudio
from torchaudio.functional import resample
from tqdm import tqdm

import torch
torch.set_num_threads(1)

from chunkformer_vpb.training.data_augment import AudioAugmenter

# ===== Config =====
# BỎ "speed" để tránh deadlock + tiết kiệm thời gian
AUG_TYPES = ["vol", "telephony", "noise"]
EXPECTED_ORIG_SR = 8000
TARGET_SR = 16000

BASE_INPUT_DIR = Path("~/dataset/data/data/processed/8khz").expanduser()
MANIFEST_INPUT_DIR = Path("~/dataset/data/manifests").expanduser()

# ===== Utils =====
def log(msg: str):
    print(msg, flush=True)

def atomic_save_wav(path: Path, wav: torch.Tensor, sr: int):
    """Ghi an toàn: write -> .tmp rồi rename (atomic)"""
    tmp = path.with_suffix(path.suffix + ".tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(tmp, wav, sr)
    os.replace(tmp, path)

def file_done(path: Path) -> bool:
    """File đã tồn tại & không rỗng (tránh ghi lại)"""
    try:
        return path.exists() and path.stat().st_size > 44
    except Exception:
        return False

# ===== Per-process state =====
_global_augmenter = None

def init_worker():
    """Mỗi process có một AudioAugmenter riêng."""
    global _global_augmenter
    _global_augmenter = AudioAugmenter(sample_rate=TARGET_SR)

# ===== Worker: process one sample =====
def process_one_sample(args):
    """
    args: (entry, dataset_name, split, output_root_str, clip_guard)
    Trả về dict:
      - "manifest": [json-able entries...]
      - "origin": 1|0
      - "aug": {aug_type: count}
      - hoặc {"error": "..."}
    """
    global _global_augmenter

    entry, dataset_name, split, output_root_str, clip_guard = args
    output_root = Path(output_root_str)

    try:
        utt_id = entry.get("utt_id", entry.get("key", "unknown_id"))
        key = entry.get("key", utt_id)
        wav_field = entry["wav"]  # bắt buộc
        input_audio_path = BASE_INPUT_DIR / dataset_name / split / "audio" / os.path.basename(wav_field)

        if not input_audio_path.exists():
            return {"error": f"Missing: {input_audio_path}"}

        # load 8k
        try:
            wav, sr = torchaudio.load(input_audio_path)
        except Exception as e:
            return {"error": f"Load failed: {input_audio_path} - {e}"}

        if sr != EXPECTED_ORIG_SR:
            return {"error": f"Invalid SR {sr} (expect {EXPECTED_ORIG_SR}): {input_audio_path}"}

        # resample -> 16k
        try:
            wav_16k = resample(wav, orig_freq=sr, new_freq=TARGET_SR)
        except Exception as e:
            return {"error": f"Resample failed: {input_audio_path} - {e}"}

        duration_sec = round(wav_16k.shape[1] / TARGET_SR, 3)

        # output dirs
        audio_base_dir = output_root / "audio" / dataset_name / split
        base_name = os.path.basename(wav_field).replace(".wav", "_16k.wav")

        # save origin (idempotent)
        origin_path = audio_base_dir / "origin" / base_name
        if not file_done(origin_path):
            atomic_save_wav(origin_path, wav_16k, TARGET_SR)

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

        # augment
        aug_counter = defaultdict(int)
        for aug_type in AUG_TYPES:
            try:
                aug_wav = _global_augmenter.apply(wav_16k.clone(), aug_type)

                # soft-clip guard: tránh peak >= 1.0 gây méo
                if clip_guard:
                    peak = float(aug_wav.abs().max())
                    if peak >= 1.0:
                        aug_wav = (aug_wav / (peak + 1e-9)) * 0.999

                aug_name = base_name.replace("_16k.wav", f"_{aug_type}.wav")
                aug_path = audio_base_dir / aug_type / aug_name
                if not file_done(aug_path):
                    atomic_save_wav(aug_path, aug_wav, TARGET_SR)

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
                # ghi tiếp các aug còn lại, không fail toàn sample
                continue

        return {
            "manifest": manifest_entries,
            "origin": 1,
            "aug": dict(aug_counter)
        }

    except Exception as e:
        # entry có thể không có "wav"
        return {"error": f"{entry.get('wav', '???')}: {str(e)}"}

# ===== Split-level processing (train/dev/test) =====
def process_dataset_split(dataset_name: str,
                          split: str,
                          mode: str,
                          limit: int,
                          log_every: int,
                          num_workers: int,
                          clip_guard: bool,
                          chunk_flush_every: int):
    """
    - Đọc manifest input: ~/dataset/data/manifests/{dataset}_{split}_manifest.json  (JSONL)
    - Xử lý song song
    - Ghi manifest output THEO CHUNK để tránh ăn RAM:
        {output_root}/manifest/{dataset}/{split}/chunks/chunk_XXXXXX.jsonl
      Sau đó merge thành:
        {output_root}/manifest/{dataset}/{dataset}_{split}_manifest.json
    """
    is_debug = (mode == "debug")
    log(f"\n🟢 [{'DEBUG' if is_debug else 'FULL'}] {dataset_name} [{split}]")

    manifest_path = MANIFEST_INPUT_DIR / f"{dataset_name}_{split}_manifest.json"
    output_root = Path("~/stt/preprocess_debug" if is_debug else "~/stt/preprocess").expanduser()
    out_manifest_dir = output_root / "manifest" / dataset_name / split
    out_manifest_dir.mkdir(parents=True, exist_ok=True)
    chunks_dir = out_manifest_dir / "chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)

    # load manifest lines
    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            if is_debug:
                lines = lines[:limit]
    except FileNotFoundError:
        log(f"❌ Manifest not found: {manifest_path}")
        return

    entries = [json.loads(line) for line in lines if line.strip()]
    n = len(entries)
    if n == 0:
        log("⚠️  Empty manifest, skip.")
        return

    log(f"🚀 Using {num_workers}/{cpu_count()} cores for {n} entries...")

    # stream kết quả & ghi manifest thành các chunk file ~chunk_flush_every
    chunk_idx = 0
    chunk_count_lines = 0
    cur_chunk_path = chunks_dir / f"chunk_{chunk_idx:06d}.jsonl"
    cur_chunk_f = open(cur_chunk_path, "w", encoding="utf-8")

    origin_counter = 0
    aug_counter = defaultdict(int)
    errors = []

    args_iter = ((entry, dataset_name, split, str(output_root), clip_guard) for entry in entries)

    with Pool(processes=num_workers, initializer=init_worker) as pool:
        processed = 0
        start_time = time.time()

        iterator = pool.imap_unordered(process_one_sample, args_iter, chunksize=16)

        # dùng tqdm nếu debug; còn full thì log nhịp nhàng (ETA)
        if is_debug:
            iterator = tqdm(iterator, total=n)

        for res in iterator:
            processed += 1

            if "error" in res:
                errors.append(res["error"])
            else:
                # ghi ngay manifest lines vào chunk hiện tại
                for m in res["manifest"]:
                    cur_chunk_f.write(json.dumps(m, ensure_ascii=False) + "\n")
                    chunk_count_lines += 1

                origin_counter += res.get("origin", 0)
                for k, v in res.get("aug", {}).items():
                    aug_counter[k] += v

            # xoay chunk file để tránh file quá to & giảm rủi ro
            if chunk_count_lines >= chunk_flush_every:
                cur_chunk_f.close()
                chunk_idx += 1
                chunk_count_lines = 0
                cur_chunk_path = chunks_dir / f"chunk_{chunk_idx:06d}.jsonl"
                cur_chunk_f = open(cur_chunk_path, "w", encoding="utf-8")

            # progress log ở chế độ full
            if not is_debug and (processed % log_every == 0 or processed == n):
                elapsed = time.time() - start_time
                rate = processed / elapsed if elapsed > 0 else 0.0
                eta = (n - processed) / rate if rate > 0 else 0.0
                pct = 100.0 * processed / n
                log(f"[{int(elapsed)}s] ⏱️ {processed}/{n} ({pct:.1f}%) - {rate:.2f} it/s - ETA {int(eta)}s")

    # close last chunk file
    try:
        cur_chunk_f.close()
    except Exception:
        pass

    # merge chunk -> manifest per-split
    out_manifest_path = out_manifest_dir / f"{dataset_name}_{split}_manifest.json"
    with open(out_manifest_path, "w", encoding="utf-8") as out_f:
        for fn in sorted(chunks_dir.glob("chunk_*.jsonl")):
            with open(fn, "r", encoding="utf-8") as f:
                for line in f:
                    out_f.write(line)

    # count lines (tổng entries)
    total_entries = 0
    with open(out_manifest_path, "r", encoding="utf-8") as f:
        for _ in f:
            total_entries += 1

    # summary
    log(f"   📄 Manifest saved: {out_manifest_path}")
    log(f"   🎧 Origin files  : {origin_counter}")
    for aug_type in AUG_TYPES:
        log(f"   🎛️  {aug_type:<10}: {aug_counter[aug_type]}")
    log(f"   📊 Total entries : {total_entries}")

    if errors:
        log(f"\n⚠️  {len(errors)} errors encountered (showing up to 10):")
        for err in errors[:10]:
            log(f"   ❌ {err}")
        if len(errors) > 10:
            log("   ...")

    gc.collect()

# ===== Dataset-level orchestrator =====
def process_dataset_big(dataset_name: str,
                        mode: str = "debug",
                        limit: int = 100,
                        num_workers: int = 8,
                        log_every: int = 2000,
                        clip_guard: bool = True,
                        chunk_flush_every: int = 5000):
    """
    Chạy lần lượt train -> dev -> test (nếu có manifest).
    """
    for split in ["train", "dev", "test"]:
        process_dataset_split(
            dataset_name=dataset_name,
            split=split,
            mode=mode,
            limit=limit,
            log_every=log_every,
            num_workers=num_workers,
            clip_guard=clip_guard,
            chunk_flush_every=chunk_flush_every,
        )

# ===== CLI =====
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True,
                        help="Big dataset name (e.g. vi_voice, viet_bud500, vietspeech)")
    parser.add_argument("--mode", choices=["debug", "full"], default="debug")
    parser.add_argument("--n", type=int, default=100, help="Number of samples per split in debug mode")
    parser.add_argument("--num_workers", type=int, default=16, help="Number of processes")
    parser.add_argument("--log_every", type=int, default=2000, help="Log every N samples in full mode")
    parser.add_argument("--no_clip_guard", action="store_true",
                        help="Disable soft-clip guard after vol augment")
    parser.add_argument("--chunk_flush_every", type=int, default=5000,
                        help="Lines per chunk file before rotating (manifest output)")
    args = parser.parse_args()

    log(f"ARGS: dataset={args.dataset} mode={args.mode} n={args.n} "
        f"num_workers={args.num_workers} log_every={args.log_every} "
        f"clip_guard={not args.no_clip_guard} chunk_flush_every={args.chunk_flush_every}")

    process_dataset_big(
        dataset_name=args.dataset,
        mode=args.mode,
        limit=args.n,
        num_workers=args.num_workers,
        log_every=args.log_every,
        clip_guard=not args.no_clip_guard,
        chunk_flush_every=args.chunk_flush_every,
    )

if __name__ == "__main__":
    main()
