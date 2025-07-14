import argparse, torch
from jiwer import wer
from .finetune_config import FinetuneConfig
from .data_loader   import get_dataloaders
from .optimizer     import build_model_and_optimizer
from .finetune_utils import _chunk_encoder_forward
from torch.nn.functional import log_softmax

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--ckpt",    required=True)
    return p.parse_args()

def main():
    args = parse_args()
    cfg = FinetuneConfig.from_yaml(args.config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, valid_loader = get_dataloaders(cfg)

    # load model
    model, tokenizer, _, _ = build_model_and_optimizer(cfg, device, total_steps=1)
    state = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(state)
    model.to(device).eval()

    tot_wer, count = 0.0, 0
    with torch.no_grad():
        for feats, feat_lens, toks, tok_lens in valid_loader:
            feats    = feats.to(device)
            feat_lens= feat_lens.to(device)

            for i in range(feats.size(0)):
                x       = feats[i].unsqueeze(0)
                x_lens  = feat_lens[i].unsqueeze(0)
                y       = toks[i].unsqueeze(0)
                y_lens  = tok_lens[i].unsqueeze(0)

                # chunk forward
                encoder_outs, encoder_mask = _chunk_encoder_forward(
                    x, model, cfg.chunk, device
                )
                encoder_lens = encoder_mask.squeeze(1).sum(1).to(torch.long)

                # greedy CTC decode
                logp = model.ctc.log_softmax(encoder_outs)
                pred_ids = logp.argmax(-1)[0].tolist()
                preds = []
                prev = None
                for pid in pred_ids:
                    if pid != prev and pid != model.blank:
                        preds.append(pid)
                    prev = pid
                pred_text = tokenizer.decode_ids(preds)

                # ref
                ref_ids  = y[0, : y_lens.item()].tolist()
                ref_text = tokenizer.decode_ids(ref_ids)

                tot_wer += wer(ref_text, pred_text)
                count   += 1

    print(f"Final Dev WER: {tot_wer/count:.2%}")

if __name__ == "__main__":
    main()
