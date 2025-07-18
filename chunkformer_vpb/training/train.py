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
    run_train(cfg_path, smoke=smoke, smoke_ratio=smoke_ratio)


def run_train(cfg_path, smoke=False, smoke_ratio=0.01):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = FinetuneConfig.from_yaml(cfg_path)

    if smoke:
        print(f"⚙️  Running in smoke mode with ratio={smoke_ratio}")
        train_loader, dev_loader = get_dataloaders_smoke(cfg, ratio=smoke_ratio)
    else:
        train_loader, dev_loader = get_dataloaders(cfg)

    total_steps = len(train_loader) * cfg.training.epochs
    model, tokenizer, optimizer, scheduler = build_model_and_optimizer(cfg, device, total_steps)
    model.to(device)

    # 👉 EVALUATE TRƯỚC TRAINING (pretrained model)
    print("\n🧪 Đánh giá mô hình trước khi fine-tune:")
    evaluate(model, tokenizer, dev_loader, cfg, device)

    global_step = 0
    num_steps_per_epoch = len(train_loader)

    for epoch in range(1, cfg.training.epochs + 1):
        model.train()
        print(f"\n🌀 Epoch {epoch} bắt đầu...")
        epoch_start_time = time.time()  # ⏱️ bắt đầu đo thời gian epoch


        for step, (feats, feat_lens, toks, tok_lens) in enumerate(train_loader, 1):
            step_start_time = time.time()  # ⏱️ bắt đầu đo thời gian step

            feats, feat_lens = feats.to(device), feat_lens.to(device)
            toks,  tok_lens  = toks.to(device),  tok_lens.to(device)

            loss, loss_ctc, loss_att = compute_loss_batch_v1(
                model, feats, feat_lens, toks, tok_lens, cfg, device
            )

            optimizer.zero_grad()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), cfg.training.max_grad_norm
            )
            optimizer.step(); scheduler.step()

            # ---- Timing ----
            step_time = time.time() - step_start_time
            remaining_steps = num_steps_per_epoch - step
            eta_epoch = remaining_steps * step_time
            eta_min, eta_sec = divmod(int(eta_epoch), 60)

            # ---- Logging ----
            lr_now = scheduler.get_last_lr()[0]
            print(f"[Epoch {epoch} Step {step}/{num_steps_per_epoch} | "
                f"Global-Step {global_step}] "
                f"loss={loss.item():.4f} (ctc={loss_ctc.item():.4f}, att={loss_att.item():.4f}) "
                f"grad={grad_norm:.2f}  lr={lr_now:.2e} "
                f"| ⏱️ {step_time:.2f}s/step - ETA: {eta_min}m{eta_sec}s")

        epoch_duration = time.time() - epoch_start_time
        ep_m, ep_s = divmod(int(epoch_duration), 60)
        print(f"✅ Epoch {epoch} hoàn tất trong {ep_m}m{ep_s}s")

        # Save checkpoint mỗi epoch
        ckpt_dir = cfg.training.checkpoint_dir
        os.makedirs(ckpt_dir, exist_ok=True)
        ckpt_path = os.path.join(ckpt_dir, f"epoch{epoch}.pt")
        torch.save(model.state_dict(), ckpt_path)
        print(f"💾 Đã lưu checkpoint: {ckpt_path}")

        # Eval sau mỗi epoch
        evaluate(model, tokenizer, dev_loader, cfg, device)


# ======== EVALUATE ========
from jiwer import wer
from chunkformer_vpb.model_utils import decode_long_form, decode_aed_long_form, get_default_args

def evaluate(model, tokenizer, loader, cfg, device, mode="ctc"):
    """
    mode: "ctc" hoặc "aed"
    """
    model.eval()
    total_wer, count = 0.0, 0
    args = get_default_args()
    args.chunk_size = cfg.chunk.chunk_size
    args.left_context_size = cfg.chunk.left_context_size
    args.right_context_size = cfg.chunk.right_context_size
    args.total_batch_duration = cfg.chunk.total_batch_duration

    char_dict = tokenizer.vocab

    with torch.no_grad():
        for feats, feat_lens, toks, tok_lens in loader:
            feats, feat_lens = feats.to(device), feat_lens.to(device)
            toks,  tok_lens  = toks.to(device),  tok_lens.to(device)

            for i in range(feats.size(0)):
                x = feats[i].unsqueeze(0)
                y = toks[i].unsqueeze(0)
                y_lens = tok_lens[i].item()

                # chuẩn hóa ref text
                ref_ids = y[0, :y_lens].tolist()
                ref_text = tokenizer.decode_ids(ref_ids)

                # CTC hoặc AED
                if mode == "ctc":
                    pred_text = decode_long_form(x, model, char_dict, args, device)
                elif mode == "aed":
                    _, pred_text = decode_aed_long_form(x, model, char_dict, args, device)
                else:
                    raise ValueError(f"Unknown mode: {mode}")
                
                # print(f"Ref: {ref_text}\nPred: {pred_text}")

                # tính WER
                total_wer += wer(ref_text.lower(), pred_text.lower())
                count += 1

    print(f"🎯 Dev WER ({mode.upper()}): {total_wer / count:.2%}")
    model.train()


# ======== MAIN ========
if __name__ == "__main__":
    train()
