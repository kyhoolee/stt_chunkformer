# def load_checkpoint(model: torch.nn.Module, path: str) -> dict:
#     if torch.cuda.is_available():
#         logging.info('Checkpoint: loading from checkpoint %s for GPU' % path)
#         checkpoint = torch.load(path, weights_only=True)
#     else:
#         logging.info('Checkpoint: loading from checkpoint %s for CPU' % path)
#         checkpoint = torch.load(path, map_location='cpu', weights_only=True)
#     missing_keys, unexpected_keys = model.load_state_dict(checkpoint, strict=False)


import torch
import logging

def load_checkpoint(model: torch.nn.Module, path: str) -> dict:
    if torch.cuda.is_available():
        logging.info(f'Checkpoint: loading from checkpoint {path} for GPU')
        checkpoint = torch.load(path, weights_only=True)
    else:
        logging.info(f'Checkpoint: loading from checkpoint {path} for CPU')
        checkpoint = torch.load(path, map_location='cpu', weights_only=True)

    # In thông tin checkpoint
    print(f"\n🧾 Loaded checkpoint from: {path}")
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
    print(f"📦 Checkpoint keys: {list(checkpoint.keys())[:5]} ... (total {len(checkpoint)})")

    # Check nếu checkpoint có head AED decoder không
    has_decoder = any(k.startswith("decoder") or ".decoder" in k for k in checkpoint.keys())
    print(f"🔍 AED decoder head included in checkpoint? {'✅ YES' if has_decoder else '❌ NO'}")

    # In tổng số tham số model
    total_param = sum(p.numel() for p in model.parameters())
    trainable_param = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"📊 Model total params: {total_param:,}, trainable: {trainable_param:,}")

    # Load với strict=False để cho phép thiếu head
    missing_keys, unexpected_keys = model.load_state_dict(checkpoint, strict=False)

    print(f"✅ Loaded state_dict with:")
    print(f"   🔺 Missing keys: {len(missing_keys)}")
    for k in missing_keys[:10]:
        print(f"     - {k}")
    if len(missing_keys) > 10:
        print("     ...")

    print(f"   ⚠️ Unexpected keys in checkpoint: {len(unexpected_keys)}")
    for k in unexpected_keys[:10]:
        print(f"     - {k}")
    if len(unexpected_keys) > 10:
        print("     ...")

    return checkpoint
