import os
import json
import re
from pathlib import Path
import torch
import pendulum
from jiwer import wer
from chunkformer_vpb.model_utils import decode_long_form, get_default_args
from chunkformer_vpb.training.finetune_config import FinetuneConfig
from chunkformer_vpb.training.data_loader import get_dataloaders_debug, get_dataloader_for_test_split
from chunkformer_vpb.training.optimizer import build_model_and_optimizer
from chunkformer_vpb.training.train import evaluate, evaluate_v0

import json
from typing import List
from jiwer import wer
from collections import defaultdict


def compare_models_wer(cache_path: str, model_ids: List[str]):
    """
    So sánh WER giữa gold và nhiều model, bao gồm cả 'pred_old'.
    In kết quả trung bình, global, theo segment, và liệt kê các sample cải thiện / kém đi.

    Args:
        cache_path: đường dẫn file JSON cache
        model_ids: danh sách các model_id cần so sánh trong sample["preds"]
    """

    with open(cache_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # data statistics 
    analyze_dataset_statistics(data)

    print(f"📊 Tổng số sample: {len(data)}")
    results = []

    all_model_ids = ["pred_old"] + model_ids

    summary = {m: [] for m in all_model_ids}
    concat_refs = {m: [] for m in all_model_ids}
    concat_hyps = {m: [] for m in all_model_ids}

    # Phân loại theo segment
    segment_summary = defaultdict(lambda: {m: [] for m in all_model_ids})
    segment_concat = defaultdict(lambda: {m: {"ref": [], "hyp": []} for m in all_model_ids})

    # So sánh cải thiện / kém đi
    improved_samples = defaultdict(list)
    worsened_samples = defaultdict(list)

    for sample in data:
        utt_id = sample["utt_id"]
        ref = sample["gold_corrected"]
        segment = sample.get("segment", "unknown")

        row = {
            "utt_id": utt_id,
            "segment": segment,
            "text_len": sample.get("text_len", -1),
            "audio_len_sec": sample.get("audio_len_sec", -1),
            "gold_corrected": ref,
            "wer": {},
            "pred_text": {}
        }

        # === pred_old ===
        hyp_old = sample.get("pred_old")
        if hyp_old:
            w = wer(ref, hyp_old)
            row["wer"]["pred_old"] = w
            row["pred_text"]["pred_old"] = hyp_old
            summary["pred_old"].append(w)
            concat_refs["pred_old"].append(ref)
            concat_hyps["pred_old"].append(hyp_old)

            segment_summary[segment]["pred_old"].append(w)
            segment_concat[segment]["pred_old"]["ref"].append(ref)
            segment_concat[segment]["pred_old"]["hyp"].append(hyp_old)
        else:
            row["wer"]["pred_old"] = None
            row["pred_text"]["pred_old"] = None

        # === Các model khác ===
        for model_id in model_ids:
            hyp = sample.get("preds", {}).get(model_id)
            if hyp:
                w = wer(ref, hyp)
                row["wer"][model_id] = w
                row["pred_text"][model_id] = hyp
                summary[model_id].append(w)
                concat_refs[model_id].append(ref)
                concat_hyps[model_id].append(hyp)

                segment_summary[segment][model_id].append(w)
                segment_concat[segment][model_id]["ref"].append(ref)
                segment_concat[segment][model_id]["hyp"].append(hyp)

                # So sánh cải thiện / kém đi
                w_old = row["wer"]["pred_old"]
                if w_old is not None:
                    if w < w_old:
                        improved_samples[model_id].append((sample, w_old, w))
                    elif w > w_old:
                        worsened_samples[model_id].append((sample, w_old, w))

            else:
                row["wer"][model_id] = None
                row["pred_text"][model_id] = None

        results.append(row)

    # === WER trung bình ===
    print("\n📈 Trung bình WER trên từng sample:")
    for m in all_model_ids:
        scores = summary[m]
        if scores:
            mean_wer = sum(scores) / len(scores) * 100
            print(f"  - {m:<15}: {mean_wer:.2f}%")
        else:
            print(f"  - {m:<15}: ❌ No data")

    # === Global WER ===
    print("\n🌐 Global WER (gộp toàn bộ text lại):")
    for m in all_model_ids:
        refs = concat_refs[m]
        hyps = concat_hyps[m]
        if refs and hyps:
            global_w = wer(" ".join(refs), " ".join(hyps)) * 100
            print(f"  - {m:<15}: {global_w:.2f}%")
        else:
            print(f"  - {m:<15}: ❌ No data")

    # === WER theo segment ===
    print("\n🧩 WER theo segment (mean / global):")
    for seg in segment_summary:
        print(f"\n🔹 Segment: {seg}")
        for m in all_model_ids:
            scores = segment_summary[seg][m]
            if scores:
                mean_wer = sum(scores) / len(scores) * 100
                refs = segment_concat[seg][m]["ref"]
                hyps = segment_concat[seg][m]["hyp"]
                global_wer_score = wer(" ".join(refs), " ".join(hyps)) * 100
                print(f"  - {m:<15}: Mean={mean_wer:.2f}% | Global={global_wer_score:.2f}%")
            else:
                print(f"  - {m:<15}: ❌ No data")

    # === In mẫu cải thiện và kém đi ===
    for m in model_ids:
        print(f"\n✅ CẢI THIỆN ({m} so với pred_old): {len(improved_samples[m])} mẫu")
        for s, w_old, w_new in improved_samples[m][:10]:  # in 10 mẫu đầu
            print(f"\n📌 {s['utt_id']} | segment={s['segment']} | len={s['text_len']} | audio={s['audio_len_sec']}s")
            print(f"  - 🟢 Gold      : {s['gold_corrected']}")
            print(f"  - 🟡 pred_old  : {s['pred_old']} ({w_old*100:.2f}%)")
            print(f"  - 🔵 {m:<10}: {s['preds'][m]} ({w_new*100:.2f}%)")

        print(f"\n⚠️  KÉM ĐI ({m} so với pred_old): {len(worsened_samples[m])} mẫu")
        for s, w_old, w_new in worsened_samples[m][:10]:
            print(f"\n📌 {s['utt_id']} | segment={s['segment']} | len={s['text_len']} | audio={s['audio_len_sec']}s")
            print(f"  - 🟢 Gold      : {s['gold_corrected']}")
            print(f"  - 🟡 pred_old  : {s['pred_old']} ({w_old*100:.2f}%)")
            print(f"  - 🔴 {m:<10}: {s['preds'][m]} ({w_new*100:.2f}%)")


def analyze_dataset_statistics(data: List[dict]):
    """
    Phân tích thống kê dataset: số lượng mẫu, phân bổ độ dài, tỉ lệ left/right, thống kê percentiles.
    """
    import numpy as np
    from collections import Counter

    total_samples = len(data)
    segments = [s.get("segment", "unknown") for s in data]
    segment_count = Counter(segments)

    text_lens = [s.get("text_len", -1) for s in data if s.get("text_len", -1) > 0]
    # audio_lens = [s.get("audio_len_sec", -1) for s in data if s.get("audio_len_sec", -1) > 0]

    def print_percentiles(values: List[int], label: str):
        p = np.percentile(values, [0, 10, 25, 50, 75, 90, 95, 99, 100])
        print(f"  - {label} phân vị:")
        print(f"    min = {p[0]:.0f} | p10 = {p[1]:.0f} | p25 = {p[2]:.0f} | median = {p[3]:.0f} | "
              f"p75 = {p[4]:.0f} | p90 = {p[5]:.0f} | p95 = {p[6]:.0f} | p99 = {p[7]:.0f} | max = {p[8]:.0f}")

    print("\n📊 Thống kê tập dữ liệu:")
    print(f"  - Tổng số mẫu       : {total_samples}")
    print(f"  - Phân bổ segment   : {dict(segment_count)}")
    print(f"  - Trung bình độ dài text: {np.mean(text_lens):.2f} từ")
    print_percentiles(text_lens, "text_len")
    # print(f"  - Trung bình audio dài : {np.mean(audio_lens):.2f}s")
    # print_percentiles(audio_lens, "audio_len_sec")

    # Thống kê riêng theo segment
    for seg in ["left", "right"]:
        seg_data = [s for s in data if s.get("segment") == seg]
        seg_text_lens = [s.get("text_len", -1) for s in seg_data if s.get("text_len", -1) > 0]
        # seg_audio_lens = [s.get("audio_len_sec", -1) for s in seg_data if s.get("audio_len_sec", -1) > 0]

        if seg_text_lens:
            print(f"\n🔹 Segment: {seg}")
            print(f"  - Số lượng mẫu       : {len(seg_data)}")
            print(f"  - Trung bình độ dài text: {np.mean(seg_text_lens):.2f} từ")
            print_percentiles(seg_text_lens, f"text_len ({seg})")

            # print(f"  - Trung bình audio dài : {np.mean(seg_audio_lens):.2f}s")
            # print_percentiles(seg_audio_lens, f"audio_len_sec ({seg})")



def extract_metadata_from_uttid(utt_id: str):
    """
    Trích thông tin segment, agent, thời gian từ utt_id.
    """
    match = re.match(
        r"E_(.*?)_D_(\d{4}-\d{2}-\d{2})_H_(\d{2})(\d{2})(\d{2})_(\d{3})_CLID_(\d+)___.*", utt_id
    )
    if not match:
        return {}
    agent_id = match.group(1)
    date = match.group(2)
    hour, minute, second = match.group(3), match.group(4), match.group(5)
    phone = match.group(7)
    call_time = f"{date} {hour}:{minute}:{second}"

    segment = "left" if "___left___" in utt_id else "right" if "___right___" in utt_id else "unknown"
    return {
        "segment": segment,
        "agent_id": agent_id,
        "call_phone": phone,
        "call_time": call_time,
    }


def update_prediction_cache(cfg_path: str,
                            cache_path: str,
                            ckpt_path,
                            model_id: str,
                            split: str = "valid",
                            device: str = "cuda",
                            overwrite: bool = False):
    """
    Dự đoán và cập nhật cache JSON với kết quả mới từ checkpoint model_id.
    """
    cfg = FinetuneConfig.from_yaml(cfg_path)


    if split in ["train", "valid"]:
        train_loader, valid_loader = get_dataloaders_debug(cfg)
        loader = train_loader if split == "train" else valid_loader
    elif split == "test":
        loader = get_dataloader_for_test_split(cfg, split_name="test")
    else:
        raise ValueError(f"Unsupported split: {split}")


    total_steps = len(loader) * cfg.training.epochs
    model, tokenizer, _, _ = build_model_and_optimizer(cfg, device, total_steps)

    if not ckpt_path or not os.path.exists(ckpt_path):
        # use base model if ckpt_path is not provided
        print(f"❌ Checkpoint not found at {ckpt_path}. Using base model instead.")
    else:
        print(f"✅ Loading checkpoint from {ckpt_path} for model_id={model_id}")
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
    
    model.to(device)
    model.eval()

    args = get_default_args()
    args.chunk_size = cfg.chunk.chunk_size
    args.left_context_size = cfg.chunk.left_context_size
    args.right_context_size = cfg.chunk.right_context_size
    args.total_batch_duration = cfg.chunk.total_batch_duration
    char_dict = tokenizer.vocab

    if os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as f:
            cache_data = {item["utt_id"]: item for item in json.load(f)}
    else:
        cache_data = {}

    with torch.no_grad():
        for batch in loader:
            feats, feat_lens, toks, tok_lens, utt_ids, golds, preds = batch
            feats = feats.to(device)

            for i in range(len(utt_ids)):
                utt_id = utt_ids[i]
                if utt_id in cache_data and not overwrite and model_id in cache_data[utt_id].get("preds", {}):
                    continue  # skip nếu đã có

                x = feats[i].unsqueeze(0)
                pred_text = decode_long_form(x, model, char_dict, args, device).strip().lower()

                if utt_id not in cache_data:
                    text = tokenizer.decode_ids(toks[i][:tok_lens[i]].tolist()).strip().lower()
                    meta = extract_metadata_from_uttid(utt_id)
                    cache_data[utt_id] = {
                        "utt_id": utt_id,
                        "text_json": text,
                        "gold_corrected": (golds[i] or "").strip().lower(),
                        "pred_old": (preds[i] or "").strip().lower(),
                        "audio_len_sec": round(feats[i].size(0) * 0.01, 2),
                        "text_len": len(text.split()),
                        "preds": {},
                        **meta
                    }


                cache_data[utt_id]["preds"][model_id] = pred_text

    # Save kết quả
    sorted_data = sorted(cache_data.values(), key=lambda x: x["utt_id"])
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(sorted_data, f, ensure_ascii=False, indent=2)
    print(f"✅ Updated prediction_cache.json with model_id={model_id}, total={len(sorted_data)} entries")



def prediction_std_test(cfg_path: str,
                            cache_path: str,
                            ckpt_path,
                            model_id: str,
                            device: str = "cuda",
                            overwrite: bool = False):
    """
    Dự đoán và cập nhật cache JSON với kết quả mới từ checkpoint model_id.
    """
    cfg = FinetuneConfig.from_yaml(cfg_path)
    loader = get_dataloader_for_test_split(cfg, split_name="train")
    


    total_steps = len(loader) * cfg.training.epochs
    model, tokenizer, _, _ = build_model_and_optimizer(cfg, device, total_steps)

    if not ckpt_path or not os.path.exists(ckpt_path):
        # use base model if ckpt_path is not provided
        print(f"❌ Checkpoint not found at {ckpt_path}. Using base model instead.")
    else:
        print(f"✅ Loading checkpoint from {ckpt_path} for model_id={model_id}")
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
    
    model.to(device)
    model.eval()

    evaluate(model, tokenizer, loader, cfg, device)



def prediction_std_test_v0(cfg_path: str,
                            cache_path: str,
                            ckpt_path,
                            model_id: str,
                            device: str = "cuda",
                            overwrite: bool = False):
    """
    Dự đoán và cập nhật cache JSON với kết quả mới từ checkpoint model_id.
    """
    cfg = FinetuneConfig.from_yaml(cfg_path)
    loader = get_dataloader_for_test_split(cfg, split_name="train")
    


    total_steps = len(loader) * cfg.training.epochs
    model, tokenizer, _, _ = build_model_and_optimizer(cfg, device, total_steps)

    if not ckpt_path or not os.path.exists(ckpt_path):
        # use base model if ckpt_path is not provided
        print(f"❌ Checkpoint not found at {ckpt_path}. Using base model instead.")
    else:
        print(f"✅ Loading checkpoint from {ckpt_path} for model_id={model_id}")
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
    
    model.to(device)
    model.eval()

    evaluate_v0(model, tokenizer, loader, cfg, device)