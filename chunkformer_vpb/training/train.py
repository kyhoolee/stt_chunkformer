import argparse, os, yaml, torch
from jiwer import wer
from .finetune_config import FinetuneConfig
from .data_loader   import get_dataloaders
from .optimizer     import build_model_and_optimizer
from .finetune_utils import compute_loss_batch, _chunk_encoder_forward

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, help="Path to finetune_config.yaml")
    return p.parse_args()

def train():
    args = parse_args()
    cfg = FinetuneConfig.from_yaml(args.config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, valid_loader = get_dataloaders(cfg)

    total_steps = len(train_loader) * cfg.training.epochs
    model, tokenizer, optim, sched = build_model_and_optimizer(
        cfg, device, total_steps
    )
    model.to(device)

    global_step = 0
    for epoch in range(1, cfg.training.epochs + 1):
        model.train()
        for feats, feat_lens, toks, tok_lens in train_loader:
            feats    = feats.to(device)
            feat_lens= feat_lens.to(device)
            toks     = toks.to(device)
            tok_lens = tok_lens.to(device)

            # compute batch loss by summing over examples
            batch_loss = 0.0
            for i in range(feats.size(0)):
                x       = feats[i].unsqueeze(0)       # [1, T, D]
                x_lens  = feat_lens[i].unsqueeze(0)   # [1]
                y       = toks[i].unsqueeze(0)        # [1, L]
                y_lens  = tok_lens[i].unsqueeze(0)    # [1]
                loss, _, _ = compute_loss_batch(model, x, x_lens, y, y_lens, cfg, device)
                batch_loss += loss
            batch_loss = batch_loss / feats.size(0)

            optim.zero_grad()
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.training.max_grad_norm)
            optim.step()
            sched.step()

            global_step += 1
            if global_step % cfg.training.log_steps == 0:
                print(f"[Epoch {epoch} Step {global_step}] loss={batch_loss.item():.4f}")

        # checkpoint per epoch
        ckpt_dir = cfg.training.checkpoint_dir
        os.makedirs(ckpt_dir, exist_ok=True)
        path = os.path.join(ckpt_dir, f"epoch{epoch}.pt")
        torch.save(model.state_dict(), path)
        print(f"Saved checkpoint: {path}")

        # eval at end of epoch
        evaluate(model, tokenizer, valid_loader, cfg, device)

def evaluate(model, tokenizer, loader, cfg, device):
    model.eval()
    tot_wer, count = 0.0, 0
    with torch.no_grad():
        for feats, feat_lens, toks, tok_lens in loader:
            feats    = feats.to(device)
            feat_lens= feat_lens.to(device)
            toks     = toks.to(device)
            tok_lens = tok_lens.to(device)

            for i in range(feats.size(0)):
                x       = feats[i].unsqueeze(0)
                x_lens  = feat_lens[i].unsqueeze(0)
                y       = toks[i].unsqueeze(0)
                y_lens  = tok_lens[i].unsqueeze(0)

                # forward chunk-encoder only to get encoder_outs
                encoder_outs, encoder_mask = _chunk_encoder_forward(
                    x, model, cfg.chunk, device
                )
                encoder_lens = encoder_mask.squeeze(1).sum(1).to(torch.long)

                # CTC greedy decode (remove blank & dup)
                # bạn có thể thay bằng get_output_with_timestamps
                logp = model.ctc.log_softmax(encoder_outs)  # [1, T, V]
                pred_ids = logp.argmax(-1)                  # [1, T]
                # remove duplicates & blanks
                preds = []
                prev = None
                for pid in pred_ids[0].tolist():
                    if pid != prev and pid != model.blank:
                        preds.append(pid)
                    prev = pid
                pred_text = tokenizer.decode_ids(preds)
                # ref text
                ref_ids = y[0, : y_lens.item()].tolist()
                ref_text = tokenizer.decode_ids(ref_ids)

                tot_wer += wer(ref_text, pred_text)
                count   += 1

    print(f"== Dev WER: {tot_wer/count:.2%} ==")
    model.train()

if __name__ == "__main__":
    train()
