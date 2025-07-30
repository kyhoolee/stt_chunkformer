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
        ref = entry.get("text", "").strip().lower()
        hyp = entry.get("base_text", "").strip().lower()

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
        for feats, feat_lens, toks, tok_lens in loader:
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
