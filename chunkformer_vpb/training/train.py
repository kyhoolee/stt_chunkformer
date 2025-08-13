#!/usr/bin/env python3
"""
fine_tune_main.py – Pipeline huấn luyện đầy đủ:
• Load config và mô hình
• Chạy training theo số epoch
• Lưu checkpoint mỗi epoch
• Tính WER trên dev set sau mỗi epoch
"""

import os, argparse, torch, yaml
import time
from jiwer import wer

from .finetune_config import FinetuneConfig
from .data_loader     import get_dataloaders, get_dataloaders_smoke
from .optimizer       import build_model_and_optimizer
from .finetune_utils  import compute_loss_batch_v1, _chunk_encoder_forward

torch.autograd.set_detect_anomaly(True)   # phát hiện NaN nếu có

# ======== ARGPARSE ========
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to finetune_config.yaml")
    parser.add_argument("--smoke", action="store_true", help="Dùng subset nhỏ để debug nhanh")
    parser.add_argument("--smoke-ratio", type=float, default=0.01, help="Tỷ lệ data dùng cho smoke test (mặc định 0.01)")
    return parser.parse_args()


# ======== TRAIN LOOP ========

def train():
    args = parse_args()
    cfg_path = args.config
    smoke = args.smoke
    smoke_ratio = args.smoke_ratio
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_train(cfg_path, smoke=smoke, smoke_ratio=smoke_ratio, device=device)

def freeze_encoder_groups(model, config):
    """
    config: FreezeConfigFT dataclass
    """
    if config.cmvn:
        for param in model.encoder.global_cmvn.parameters():
            param.requires_grad = False

    if config.subsampling:
        for param in model.encoder.embed.parameters():
            param.requires_grad = False

    if config.post_embed_norm:
        for param in model.encoder.after_norm.parameters():
            param.requires_grad = False

    if config.encoder_layers > 0:
        freeze_n = config.encoder_layers
        for i, layer in enumerate(model.encoder.encoders):
            if i < freeze_n:
                for param in layer.parameters():
                    param.requires_grad = False

    if config.ctc:
        for param in model.encoder.ctc.parameters():
            param.requires_grad = False


# ======== RUN TRAIN ========

def run_train(cfg_path, smoke=False, smoke_ratio=0.01,
              eval_train=False, eval_ratio=0.1,
              device="cpu", resume_ckpt_path=None):

    cfg = FinetuneConfig.from_yaml(cfg_path)

    # === Load dataloader ===
    if smoke:
        print(f"⚙️  Running in smoke mode with ratio={smoke_ratio}")
        train_loader, dev_loader = get_dataloaders_smoke(cfg, ratio=smoke_ratio)
    else:
        train_loader, dev_loader = get_dataloaders(cfg)

    total_steps = len(train_loader) * cfg.training.epochs
    model, tokenizer, optimizer, scheduler = build_model_and_optimizer(cfg, device, total_steps)

    # === Resume checkpoint nếu có ===
    if resume_ckpt_path and os.path.exists(resume_ckpt_path):
        print(f"♻️ Resume training từ checkpoint: {resume_ckpt_path}")
        ckpt = torch.load(resume_ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = ckpt.get("epoch", 0) + 1
        global_step = ckpt.get("global_step", 0)
    else:
        print(f"🔰 Khởi tạo từ đầu (pretrained model)")
        start_epoch = 1
        global_step = 0

    model.to(device)

    # === Freeze encoder nếu cần ===
    if hasattr(cfg, "freeze") and cfg.freeze is not None:
        freeze_encoder_groups(model, cfg.freeze)

    # === Thông tin thiết bị ===
    print(f"💻 Sử dụng thiết bị: {device}")
    model_device = next(model.parameters()).device
    if model_device != device:
        print(f"⚠️ Mô hình không ở thiết bị đúng! ({model_device})")
    else:
        print(f"✅ Mô hình đã được chuyển sang đúng thiết bị: {device}")

    # === Evaluate trước training ===
    print("\n🧪 Đánh giá mô hình trước khi fine-tune:")
    if dev_loader is not None:
        evaluate(model, tokenizer, dev_loader, cfg, device)
    else:
        print("⚠️ Không có dữ liệu dev để đánh giá!")

    if eval_train:
        print("\n🧪 Đánh giá mô hình trên tập train:")
        evaluate(model, tokenizer, train_loader, cfg, device, eval_ratio=eval_ratio)

    # === Train loop ===
    num_steps_per_epoch = len(train_loader)

    for epoch in range(start_epoch, cfg.training.epochs + 1):
        model.train()
        print(f"\n🌀 Epoch {epoch} bắt đầu...")
        epoch_start_time = time.time()

        for step, (feats, feat_lens, toks, tok_lens) in enumerate(train_loader, 1):
            step_start_time = time.time()

            feats, feat_lens = feats.to(device), feat_lens.to(device)
            toks,  tok_lens  = toks.to(device),  tok_lens.to(device)

            loss, loss_ctc, loss_att = compute_loss_batch_v1(
                model, feats, feat_lens, toks, tok_lens, cfg, device
            )

            optimizer.zero_grad()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.training.max_grad_norm)
            optimizer.step()
            scheduler.step()
            global_step += 1

            step_time = time.time() - step_start_time
            remaining_steps = num_steps_per_epoch - step
            eta_epoch = remaining_steps * step_time
            eta_min, eta_sec = divmod(int(eta_epoch), 60)

            lr_now = scheduler.get_last_lr()[0]
            print(f"[Epoch {epoch} Step {step}/{num_steps_per_epoch} | "
                  f"Global-Step {global_step}] "
                  f"loss={loss.item():.4f} (ctc={loss_ctc.item():.4f}, att={loss_att.item():.4f}) "
                  f"grad={grad_norm:.2f}  lr={lr_now:.2e} "
                  f"| ⏱️ {step_time:.2f}s/step - ETA: {eta_min}m{eta_sec}s")

        epoch_duration = time.time() - epoch_start_time
        ep_m, ep_s = divmod(int(epoch_duration), 60)
        print(f"✅ Epoch {epoch} hoàn tất trong {ep_m}m{ep_s}s")

        # === Save checkpoint (đầy đủ state) ===
        ckpt_dir = cfg.training.checkpoint_dir
        os.makedirs(ckpt_dir, exist_ok=True)
        ckpt_path = os.path.join(ckpt_dir, f"epoch{epoch}.pt")
        torch.save({
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "epoch": epoch,
            "global_step": global_step,
        }, ckpt_path)
        print(f"💾 Đã lưu checkpoint: {ckpt_path}")

        # === Evaluate dev ===
        evaluate(model, tokenizer, dev_loader, cfg, device)


# ======== EVALUATE BUCKETS ========
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Eval WER + CER + Char Confusions theo SNR bucket cho manifest JSON array.

Usage:
  python eval_report.py \
    --meta manifest_user_only.with_snr.json \
    --bucket-field snr_bucket \
    --ref-field text \
    --hyp-field base_text \
    --exclude-buckets noisy \
    --conf-topn 20 \
    --cer-strip-spaces

Requires:
  pip install jiwer
"""
import json
import math
import re
import argparse
from collections import defaultdict, Counter
from typing import List, Tuple, Dict, Any, Optional
from jiwer import wer as jiwer_wer

# =============== Text normalization (nhẹ) ===============
def _norm(s: str) -> str:
    s = (s or "").lower().strip()
    s = re.sub(r"\s+", " ", s)
    return s

# =============== CER & Alignment utils =================
def _cer_value(ref: str, hyp: str, strip_spaces: bool = True) -> float:
    """
    CER = edit_distance(chars) / len(ref_chars). Khoảng trắng mặc định bỏ.
    """
    r = ref.replace(" ", "") if strip_spaces else ref
    h = hyp.replace(" ", "") if strip_spaces else hyp
    if len(r) == 0:
        return 1.0 if len(h) > 0 else 0.0
    _, dist = _align_chars(r, h)  # dist = edit distance
    return dist / len(r)

def _align_chars(ref: str, hyp: str) -> Tuple[List[Tuple[Optional[str], Optional[str]]], int]:
    """
    Trả về:
      - path: danh sách cặp (r_char | None, h_char | None)
              None ở 1 phía thể hiện delete/insert
      - dist: Levenshtein distance (S + D + I)
    Thuật toán: DP chuẩn (O(len(ref)*len(hyp))) với truy vết.
    """
    R, H = list(ref), list(hyp)
    n, m = len(R), len(H)
    # dp[i][j] = distance for R[:i], H[:j]
    dp = [[0]*(m+1) for _ in range(n+1)]
    bt = [[(0,0)]*(m+1) for _ in range(n+1)]  # backtrace: from where

    for i in range(1, n+1):
        dp[i][0] = i
        bt[i][0] = (i-1, 0)
    for j in range(1, m+1):
        dp[0][j] = j
        bt[0][j] = (0, j-1)

    for i in range(1, n+1):
        ri = R[i-1]
        for j in range(1, m+1):
            hj = H[j-1]
            cost_sub = 0 if ri == hj else 1
            # three options
            a = dp[i-1][j] + 1        # deletion (ri -> ε)
            b = dp[i][j-1] + 1        # insertion (ε -> hj)
            c = dp[i-1][j-1] + cost_sub  # substitution / match
            best = c
            prev = (i-1, j-1)
            if a < best:
                best, prev = a, (i-1, j)
            if b < best:
                best, prev = b, (i, j-1)
            dp[i][j] = best
            bt[i][j] = prev

    # backtrace path
    path: List[Tuple[Optional[str], Optional[str]]] = []
    i, j = n, m
    while i > 0 or j > 0:
        pi, pj = bt[i][j]
        if pi == i-1 and pj == j-1:
            # sub or match
            path.append((R[i-1], H[j-1]))
        elif pi == i-1 and pj == j:
            # deletion (ri -> ε)
            path.append((R[i-1], None))
        elif pi == i and pj == j-1:
            # insertion (ε -> hj)
            path.append((None, H[j-1]))
        else:
            # should not happen
            break
        i, j = pi, pj

    path.reverse()
    return path, dp[n][m]

# =============== Pretty helpers =========================
def _fmt_pct(x: float) -> str:
    return f"{x:.2%}"

def _bucket_order(keys):
    pref = {"clean": 0, "mid": 1, "noisy": 2, "unknown": 3}
    return sorted(keys, key=lambda k: (pref.get(k, 99), k))

# =============== Core evaluation ========================
def evaluate_detail_bucket(
    meta_path: str,
    bucket_field: str = "snr_bucket",
    ref_field: str = "text",
    hyp_field: str = "base_text",
    exclude_buckets: Optional[List[str]] = None,
    cer_strip_spaces: bool = True,
    conf_topn: int = 20,
    include_insert_delete: bool = True,
) -> Dict[str, Any]:
    with open(meta_path, "r", encoding="utf-8") as f:
        entries = json.load(f)
    if not isinstance(entries, list):
        raise ValueError("Manifest must be a JSON array.")

    if exclude_buckets is None:
        exclude_buckets = []

    # overall accumulators
    total_wer = total_cer = 0.0
    count = 0
    all_refs: List[str] = []
    all_hyps: List[str] = []

    # per-bucket
    b_sum_wer = defaultdict(float)
    b_sum_cer = defaultdict(float)
    b_count = defaultdict(int)
    b_refs = defaultdict(list)
    b_hyps = defaultdict(list)
    # confusion counters: Counter[(ref_char or '␀', hyp_char or '␀')]
    b_conf = defaultdict(Counter)
    # error totals per bucket for percentage denominator
    b_err_total = defaultdict(int)

    print(f"======>>>>>> Total entries: {len(entries)}")

    for e in entries:
        ref = _norm(e.get(ref_field, ""))
        hyp = _norm(e.get(hyp_field, ""))
        if not ref or not hyp:
            continue
        b = (e.get(bucket_field) or "unknown")
        if b in exclude_buckets:
            continue

        s_wer = jiwer_wer(ref, hyp)
        s_cer = _cer_value(ref, hyp, strip_spaces=cer_strip_spaces)

        total_wer += s_wer
        total_cer += s_cer
        count += 1
        all_refs.append(ref)
        all_hyps.append(hyp)

        b_sum_wer[b] += s_wer
        b_sum_cer[b] += s_cer
        b_count[b] += 1
        b_refs[b].append(ref)
        b_hyps[b].append(hyp)

        # confusions
        r = ref.replace(" ", "") if cer_strip_spaces else ref
        h = hyp.replace(" ", "") if cer_strip_spaces else hyp
        path, dist = _align_chars(r, h)
        b_err_total[b] += dist
        for rc, hc in path:
            if rc == hc:
                continue  # match
            ref_c = rc if rc is not None else "␀"  # epsilon
            hyp_c = hc if hc is not None else "␀"
            if not include_insert_delete and ("␀" in (ref_c, hyp_c)):
                continue
            b_conf[b][(ref_c, hyp_c)] += 1


        # print debug 
        if count % 10 == 0:
            print(f"Processed {count} samples: {b} | WER: {_fmt_pct(s_wer)} | CER: {_fmt_pct(s_cer)}")
    print("======>>>>>> Finished processing entries.")            

    if count == 0:
        return {"error": "No valid samples after filtering."}

    # overall metrics
    overall = {
        "num_samples": count,
        "avg_wer": total_wer / count,
        "avg_cer": total_cer / count,
        "global_wer": jiwer_wer(all_refs, all_hyps),
        "global_cer": _cer_value("".join(all_refs), "".join(all_hyps), cer_strip_spaces),
    }

    print(f"====== OVERALL =====")
    print(json.dumps(overall, ensure_ascii=False, indent=2))

    # per-bucket metrics

    ## print debug 
    by_bucket = {}
    for b in _bucket_order(b_count.keys()):
        n = b_count[b]
        if n == 0:
            continue
        avg_wer = b_sum_wer[b] / n
        avg_cer = b_sum_cer[b] / n
        glb_wer = jiwer_wer(b_refs[b], b_hyps[b])
        glb_cer = _cer_value("".join(b_refs[b]), "".join(b_hyps[b]), cer_strip_spaces)

        # top confusions
        conf_counter = b_conf[b]
        err_total = max(1, b_err_total[b])  # avoid div0
        top_items = conf_counter.most_common(conf_topn)
        top_conf = [
            {
                "ref": k[0],
                "hyp": k[1],
                "count": v,
                "ratio": v / err_total,  # tỉ lệ trên tổng lỗi (S+D+I) của bucket
            }
            for k, v in top_items
        ]

        by_bucket[b] = {
            "num_samples": n,
            "avg_wer": avg_wer,
            "avg_cer": avg_cer,
            "global_wer": glb_wer,
            "global_cer": glb_cer,
            "errors_total": b_err_total[b],
            "top_confusions": top_conf,
        }

        print(f"====== BUCKET [{b}] =====")
        print(json.dumps(by_bucket[b], ensure_ascii=False, indent=2))

    return {"overall": overall, "by_bucket": by_bucket}

# =============== Pretty print ===========================
def print_report(report: Dict[str, Any], conf_topn: int):
    if "error" in report:
        print(f"⚠️ {report['error']}")
        return
    ov = report["overall"]
    print("===== OVERALL =====")
    print(f"📊 Tổng số mẫu: {ov['num_samples']}")
    print(f"🎯 WER trung bình: {_fmt_pct(ov['avg_wer'])}")
    print(f"🅲 CER trung bình: {_fmt_pct(ov['avg_cer'])}")
    print(f"🌐 WER toàn cục : {_fmt_pct(ov['global_wer'])}")
    print(f"🌐 CER toàn cục : {_fmt_pct(ov['global_cer'])}")
    print()

    print("===== BY SNR BUCKET =====")
    for b, info in report["by_bucket"].items():
        print(f"[{b}]")
        print(f"  📦 samples        : {info['num_samples']}")
        print(f"  🎯 sample-avg WER : {_fmt_pct(info['avg_wer'])}")
        print(f"  🅲 sample-avg CER : {_fmt_pct(info['avg_cer'])}")
        print(f"  🌐 global WER     : {_fmt_pct(info['global_wer'])}")
        print(f"  🌐 global CER     : {_fmt_pct(info['global_cer'])}")
        print(f"  ❌ total edits    : {info['errors_total']}")
        print(f"  --- Top {conf_topn} char confusions (ref → hyp) ---")
        for i, it in enumerate(info["top_confusions"], 1):
            ref_c = it["ref"]
            hyp_c = it["hyp"]
            # hiển thị epsilon đẹp
            ref_s = ref_c if ref_c != "␀" else "∅"
            hyp_s = hyp_c if hyp_c != "␀" else "∅"
            print(f"   {i:2d}. '{ref_s}' → '{hyp_s}': {it['count']} ({_fmt_pct(it['ratio'])})")
        print()

# # =============== CLI ===================================
# def main():
#     ap = argparse.ArgumentParser(description="Eval WER/CER + char confusions per SNR bucket (JSON array manifest).")
#     ap.add_argument("--meta", required=True, help="Path to JSON array manifest")
#     ap.add_argument("--bucket-field", default="snr_bucket")
#     ap.add_argument("--ref-field", default="text")
#     ap.add_argument("--hyp-field", default="base_text")
#     ap.add_argument("--exclude-buckets", default="", help="Comma-separated buckets to exclude, e.g., 'noisy,unknown'")
#     ap.add_argument("--cer-strip-spaces", action="store_true", help="Compute CER after removing spaces")
#     ap.add_argument("--conf-topn", type=int, default=20, help="Top-N confusions to display per bucket")
#     ap.add_argument("--no-insert-delete", action="store_true",
#                     help="If set, exclude insertions/deletions (ε) from confusion table; only substitutions kept")
#     ap.add_argument("--out-json", default="", help="Optional: write full report JSON to this path")
#     args = ap.parse_args()

#     excl = [b.strip() for b in args.exclude_buckets.split(",") if b.strip()]
#     report = evaluate_report(
#         meta_path=args.meta,
#         bucket_field=args.bucket_field,
#         ref_field=args.ref_field,
#         hyp_field=args.hyp_field,
#         exclude_buckets=excl,
#         cer_strip_spaces=args.cer_strip_spaces,
#         conf_topn=args.conf_topn,
#         include_insert_delete=not args.no_insert_delete,
#     )
#     print_report(report, conf_topn=args.conf_topn)
#     if args.out_json:
#         with open(args.out_json, "w", encoding="utf-8") as f:
#             json.dump(report, f, ensure_ascii=False, indent=2)

# if __name__ == "__main__":
#     main()


# ======== EVALUATE BUCKETS ========

#!/usr/bin/env python3
import json
import re
from collections import defaultdict
from typing import List, Dict, Any, Optional
from jiwer import wer
import editdistance

# ---------- text normalization (nhẹ) ----------
def _norm(s: str) -> str:
    s = (s or "").lower().strip()
    # chuẩn hoá khoảng trắng
    s = re.sub(r"\s+", " ", s)
    return s

def _cer(ref: str, hyp: str, strip_spaces: bool = True) -> float:
    """
    CER: (S + D + I) / |ref|
    - strip_spaces=True: loại bỏ khoảng trắng trước khi so ký tự (khuyến nghị cho TViệt)
    """
    if strip_spaces:
        ref = ref.replace(" ", "")
        hyp = hyp.replace(" ", "")
    ref_chars = list(ref)
    hyp_chars = list(hyp)
    dist = editdistance.eval(ref_chars, hyp_chars)
    return dist / max(1, len(ref_chars))

def evaluate_from_meta_by_bucket(
    meta_path: str,
    bucket_field: str = "snr_bucket",
    ref_field: str = "text",
    hyp_field: str = "base_text",
    exclude_buckets: Optional[List[str]] = None,
    cer_strip_spaces: bool = True,
):
    """
    Đọc manifest JSON array và in:
      - OVERALL: sample-avg WER/CER + global WER/CER
      - BY BUCKET (snr): sample-avg & global cho từng bucket

    Args:
        meta_path: đường dẫn manifest JSON (array)
        bucket_field: trường bucket (vd 'snr_bucket')
        ref_field: trường ref
        hyp_field: trường hyp (ASR output)
        exclude_buckets: danh sách bucket cần loại (vd ['noisy'])
        cer_strip_spaces: CER tính theo ký tự sau khi bỏ khoảng trắng
    """
    with open(meta_path, "r", encoding="utf-8") as f:
        entries: List[Dict[str, Any]] = json.load(f)

    if exclude_buckets is None:
        exclude_buckets = []

    # Tổng thể
    total_wer = 0.0
    total_cer = 0.0
    count = 0
    all_refs: List[str] = []
    all_hyps: List[str] = []

    # Theo bucket
    bucket_refs: Dict[str, List[str]] = defaultdict(list)
    bucket_hyps: Dict[str, List[str]] = defaultdict(list)
    bucket_sum_wer: Dict[str, float] = defaultdict(float)
    bucket_sum_cer: Dict[str, float] = defaultdict(float)
    bucket_count: Dict[str, int] = defaultdict(int)

    for e in entries:
        ref = _norm(e.get(ref_field, ""))
        hyp = _norm(e.get(hyp_field, ""))
        if not ref or not hyp:
            continue

        b = (e.get(bucket_field) or "unknown")
        if b in exclude_buckets:
            continue

        s_wer = wer(ref, hyp)
        s_cer = _cer(ref, hyp, strip_spaces=cer_strip_spaces)

        total_wer += s_wer
        total_cer += s_cer
        count += 1
        all_refs.append(ref)
        all_hyps.append(hyp)

        bucket_sum_wer[b] += s_wer
        bucket_sum_cer[b] += s_cer
        bucket_count[b] += 1
        bucket_refs[b].append(ref)
        bucket_hyps[b].append(hyp)

    if count == 0:
        print("⚠️ No valid samples after filtering.")
        return

    avg_wer = total_wer / count
    avg_cer = total_cer / count
    global_wer = wer(all_refs, all_hyps)
    # global CER: nối chuỗi (hoặc có thể gộp theo list ký tự)
    global_cer = _cer("".join(all_refs), "".join(all_hyps), strip_spaces=cer_strip_spaces)

    print("===== OVERALL =====")
    print(f"📊 Tổng số mẫu: {count}")
    print(f"🎯 WER trung bình: {avg_wer:.2%}")
    print(f"🅲 CER trung bình: {avg_cer:.2%}")
    print(f"🌐 WER toàn cục : {global_wer:.2%}")
    print(f"🌐 CER toàn cục : {global_cer:.2%}")
    print()

    # Thứ tự in bucket: clean, mid, noisy, unknown, rồi alphabet
    preferred = {"clean": 0, "mid": 1, "noisy": 2, "unknown": 3}
    buckets = list(bucket_count.keys())
    buckets.sort(key=lambda x: (preferred.get(x, 99), x))

    print("===== BY SNR BUCKET =====")
    for b in buckets:
        n = bucket_count[b]
        if n == 0:
            continue
        b_avg_wer = bucket_sum_wer[b] / n
        b_avg_cer = bucket_sum_cer[b] / n
        b_glb_wer = wer(bucket_refs[b], bucket_hyps[b])
        b_glb_cer = _cer("".join(bucket_refs[b]), "".join(bucket_hyps[b]), strip_spaces=cer_strip_spaces)
        print(f"[{b}]")
        print(f"  📦 samples        : {n}")
        print(f"  🎯 sample-avg WER : {b_avg_wer:.2%}")
        print(f"  🅲 sample-avg CER : {b_avg_cer:.2%}")
        print(f"  🌐 global WER     : {b_glb_wer:.2%}")
        print(f"  🌐 global CER     : {b_glb_cer:.2%}")
        print()





# ======== EVALUATE ========

import json
from jiwer import wer
from pathlib import Path

def evaluate_from_meta(meta_path: str):
    with open(meta_path, "r", encoding="utf-8") as f:
        entries = json.load(f)

    total_wer = 0.0
    count = 0
    all_refs = []
    all_preds = []

    for entry in entries:
        try: 
            ref = entry.get("text", "").strip().lower()
            hyp = entry.get("base_text", "").strip().lower()
        except Exception as e:
            print(f"⚠️ Lỗi khi đọc entry {entry}: {e}")
            # print stack trace
            import traceback
            traceback.print_exc()
            # print value:
            print(f"  - ref: {ref}")
            print(f"  - hyp: {hyp}")
            exit
        if not ref or not hyp:
            continue

        sample_wer = wer(ref, hyp)
        total_wer += sample_wer
        count += 1

        all_refs.append(ref)
        all_preds.append(hyp)

    if count == 0:
        print("⚠️ No valid samples found.")
        return

    avg_wer = total_wer / count
    global_wer = wer(all_refs, all_preds)

    print(f"📊 Tổng số mẫu: {count}")
    print(f"🎯 WER trung bình (sample avg): {avg_wer:.2%}")
    print(f"🌐 WER toàn cục   (global):     {global_wer:.2%}")



from jiwer import wer
from chunkformer_vpb.model_utils import decode_long_form, decode_aed_long_form, get_default_args
from jiwer import wer

def evaluate(model, tokenizer, loader, cfg, device, mode="ctc", eval_ratio=1.0):
    if loader is None:
        print("🚫 No eval data found. Skipping evaluation.")
        return

    model.eval()
    total_wer, count = 0.0, 0
    all_refs, all_preds = [], []

    args = get_default_args()
    args.chunk_size = cfg.chunk.chunk_size
    args.left_context_size = cfg.chunk.left_context_size
    args.right_context_size = cfg.chunk.right_context_size
    args.total_batch_duration = cfg.chunk.total_batch_duration

    char_dict = tokenizer.vocab

    max_samples = int(len(loader.dataset) * eval_ratio)
    processed_samples = 0

    start_eval_time = time.time()  # ⏱️ Tổng thời gian evaluate
    total_decode_time = 0.0

    with torch.no_grad():
        for loader_item in loader:
            feats, feat_lens, toks, tok_lens = loader_item[0:4]  # Lấy 4 phần đầu tiên
            feats, feat_lens = feats.to(device), feat_lens.to(device)
            toks,  tok_lens  = toks.to(device),  tok_lens.to(device)

            for i in range(feats.size(0)):
                if processed_samples >= max_samples:
                    break

                x = feats[i].unsqueeze(0)
                y = toks[i].unsqueeze(0)
                y_lens = tok_lens[i].item()

                ref_ids = y[0, :y_lens].tolist()
                ref_text = tokenizer.decode_ids(ref_ids)

                decode_start = time.time()
                if mode == "ctc":
                    pred_text = decode_long_form(x, model, char_dict, args, device)
                elif mode == "aed":
                    _, pred_text = decode_aed_long_form(x, model, char_dict, args, device)
                else:
                    raise ValueError(f"Unknown mode: {mode}")
                decode_duration = time.time() - decode_start
                total_decode_time += decode_duration

                ref_text = ref_text.lower()
                pred_text = pred_text.lower()

                total_wer += wer(ref_text, pred_text)
                count += 1
                processed_samples += 1

                all_refs.append(ref_text)
                all_preds.append(pred_text)

            if processed_samples >= max_samples:
                break

    total_eval_time = time.time() - start_eval_time

    if count == 0:
        print("⚠️ No samples evaluated.")
        return

    avg_wer = total_wer / count
    global_wer = wer(all_refs, all_preds)
    avg_decode_time = total_decode_time / count

    print(f"🎯 Dev WER ({mode.upper()}): {avg_wer:.2%}")
    print(f"🌐 Global WER           : {global_wer:.2%}")
    print(f"🕒 Evaluate time: {total_eval_time:.2f}s "
          f"(avg decode/sample: {avg_decode_time:.2f}s)")
    model.train()


####################################################

import csv
import torch
import time
from jiwer import wer
from chunkformer_vpb.model_utils import decode_long_form, decode_aed_long_form, get_default_args

def evaluate_debug(model, tokenizer, loader, cfg, device,
                   output_csv="debug_eval_output.csv", mode="ctc", eval_ratio=1.0):
    """
    Đánh giá chi tiết để debug sự khác biệt giữa:
    - text (label json gốc)
    - gold_corrected (label thủ công)
    - pred_old (output cũ của model)
    - pred_new (output mới từ mô hình hiện tại)
    """

    if loader is None:
        print("🚫 No eval data found. Skipping.")
        return

    model.eval()
    args = get_default_args()
    args.chunk_size = cfg.chunk.chunk_size
    args.left_context_size = cfg.chunk.left_context_size
    args.right_context_size = cfg.chunk.right_context_size
    args.total_batch_duration = cfg.chunk.total_batch_duration

    char_dict = tokenizer.vocab
    max_samples = int(len(loader.dataset) * eval_ratio)
    processed_samples = 0
    total_decode_time = 0.0

    rows = []

    with torch.no_grad():
        for batch in loader:
            feats, feat_lens, toks, tok_lens, utt_ids, golds, preds = batch
            feats, feat_lens = feats.to(device), feat_lens.to(device)
            toks, tok_lens = toks.to(device), tok_lens.to(device)

            for i in range(feats.size(0)):
                if processed_samples >= max_samples:
                    break

                x = feats[i].unsqueeze(0)
                y = toks[i].unsqueeze(0)
                y_lens = tok_lens[i].item()

                # text từ json (label huấn luyện cũ)
                ref_ids = y[0, :y_lens].tolist()
                text_json = tokenizer.decode_ids(ref_ids).lower()

                # decode từ model hiện tại
                decode_start = time.time()
                if mode == "ctc":
                    pred_new = decode_long_form(x, model, char_dict, args, device)
                elif mode == "aed":
                    _, pred_new = decode_aed_long_form(x, model, char_dict, args, device)
                else:
                    raise ValueError(f"Unknown mode: {mode}")
                total_decode_time += time.time() - decode_start

                pred_new = pred_new.lower()
                gold = (golds[i] or "").lower()
                pred_old = (preds[i] or "").lower()
                utt_id = utt_ids[i]

                r = {
                    "utt_id": utt_id,
                    "text_json": text_json,
                    "gold_corrected": gold,
                    "pred_old": pred_old,
                    "pred_new": pred_new,
                    "wer_json_vs_new": wer(text_json, pred_new),
                    "wer_gold_vs_new": wer(gold, pred_new) if gold else None,
                    "wer_gold_vs_old": wer(gold, pred_old) if gold and pred_old else None,
                }

                rows.append(r)
                print(f"🔍 Sample {processed_samples + 1}/{max_samples} - "
                      f"utt_id: {utt_id}, WER(json vs new): {r['wer_json_vs_new']:.2%}, "
                      f"WER(gold vs new): {r['wer_gold_vs_new']:.2%} "
                      f"WER(gold vs old): {r['wer_gold_vs_old']:.2%}")
                print(f"  - text_json: {text_json}")
                print(f"  - gold_corrected: {gold}")
                print(f"  - pred_old: {pred_old}")
                print(f"  - pred_new: {pred_new}")
                print("-" * 80)



                processed_samples += 1
            if processed_samples >= max_samples:
                break

    # Ghi ra CSV để phân tích
    # create folder 
    os.makedirs(cfg.training.checkpoint_dir, exist_ok=True)
    output_path = os.path.join(cfg.training.checkpoint_dir, output_csv)
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"✅ Saved debug result to {output_path}")
    print(f"🔍 Evaluated {processed_samples} samples")
    print(f"🕒 Avg decode/sample: {total_decode_time / processed_samples:.2f}s")



# ======== MAIN ========
if __name__ == "__main__":
    train()
