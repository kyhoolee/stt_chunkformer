#!/usr/bin/env python3
"""
fine_tune_main.py – Pipeline huấn luyện đầy đủ:
• Load config và mô hình
• Chạy training theo số epoch
• Lưu checkpoint mỗi epoch
• Tính WER trên dev set sau mỗi epoch
"""

import os, argparse, torch, yaml
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

    return parser.parse_args()

# ======== TRAIN LOOP ========

def train():
    args = parse_args()
    cfg = FinetuneConfig.from_yaml(args.config)
    smoke = args.smoke
    run_train(cfg, smoke)


def run_train(cfg, smoke=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # train_loader, dev_loader = get_dataloaders(cfg)
    # train_loader, dev_loader = get_dataloaders_smoke(cfg, ratio=0.01)
    if smoke:
        train_loader, dev_loader = get_dataloaders_smoke(cfg, ratio=0.01)
    else:
        train_loader, dev_loader = get_dataloaders(cfg)


    total_steps = len(train_loader) * cfg.training.epochs
    model, tokenizer, optimizer, scheduler = build_model_and_optimizer(cfg, device, total_steps)
    model.to(device)

    # 👉 EVALUATE TRƯỚC TRAINING (pretrained model)
    print("\n🧪 Đánh giá mô hình trước khi fine-tune:")
    evaluate(model, tokenizer, dev_loader, cfg, device)

    global_step = 0
    for epoch in range(1, cfg.training.epochs + 1):
        model.train()
        print(f"\n🌀 Epoch {epoch} bắt đầu...")

        for step, (feats, feat_lens, toks, tok_lens) in enumerate(train_loader, 1):
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
            global_step += 1

            # ---- Logging ----
            if global_step % cfg.training.log_steps == 0:
                lr_now = scheduler.get_last_lr()[0]
                print(f"[Epoch {epoch} | Step {global_step}] "
                      f"loss={loss.item():.4f} (ctc={loss_ctc.item():.4f}, att={loss_att.item():.4f}) "
                      f"grad={grad_norm:.2f}  lr={lr_now:.2e}")

        # Save checkpoint mỗi epoch
        ckpt_dir = cfg.training.checkpoint_dir
        os.makedirs(ckpt_dir, exist_ok=True)
        ckpt_path = os.path.join(ckpt_dir, f"epoch{epoch}.pt")
        torch.save(model.state_dict(), ckpt_path)
        print(f"💾 Đã lưu checkpoint: {ckpt_path}")

        # Eval sau mỗi epoch
        evaluate(model, tokenizer, dev_loader, cfg, device)


# ======== EVALUATE ========
def evaluate(model, tokenizer, loader, cfg, device):
    model.eval()
    tot_wer, count = 0.0, 0

    with torch.no_grad():
        for feats, feat_lens, toks, tok_lens in loader:
            feats, feat_lens = feats.to(device), feat_lens.to(device)
            toks,  tok_lens  = toks.to(device),  tok_lens.to(device)

            for i in range(feats.size(0)):
                x = feats[i].unsqueeze(0)
                x_lens = feat_lens[i].unsqueeze(0)
                y = toks[i].unsqueeze(0)
                y_lens = tok_lens[i].unsqueeze(0)

                # encoder forward
                enc_out, enc_mask = _chunk_encoder_forward(x, model, cfg.chunk, device)
                enc_len = enc_mask.squeeze(1).sum(1).long()

                logp = model.ctc.log_softmax(enc_out)  # [1, T, V]
                pred_ids = logp.argmax(dim=-1)[0].tolist()

                # decode: remove blanks & dups
                pred_seq, prev = [], None
                for pid in pred_ids:
                    if pid != model.blank and pid != prev:
                        pred_seq.append(pid)
                    prev = pid
                pred_text = tokenizer.decode_ids(pred_seq)

                # ground truth
                ref_ids = y[0, :y_lens.item()].tolist()
                ref_text = tokenizer.decode_ids(ref_ids)

                tot_wer += wer(ref_text, pred_text)
                count += 1

    print(f"🎯 Dev WER: {tot_wer / count:.2%}")
    model.train()

# ======== MAIN ========
if __name__ == "__main__":
    train()
