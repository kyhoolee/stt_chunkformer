# compare_decoder.py

import torch
from wenet.transformer.decoder import BiTransformerDecoder

def instantiate_and_compare(checkpoint_path):
    # 1) Load checkpoint
    ckpt = torch.load(f"{checkpoint_path}/pytorch_model.bin", map_location="cpu")

    # 2) Infer hyperparameters from checkpoint
    vocab_size = ckpt["decoder.left_decoder.embed.0.weight"].size(0)
    d_model    = ckpt["decoder.left_decoder.after_norm.weight"].size(0)
    # determine number of decoder blocks
    block_ids  = {int(k.split('.')[3]) for k in ckpt if k.startswith("decoder.left_decoder.decoders.")}
    num_blocks = max(block_ids) + 1

    # 3) Instantiate BiTransformerDecoder
    decoder = BiTransformerDecoder(
        vocab_size=vocab_size,
        encoder_output_size=d_model,
        attention_heads=8,
        linear_units=2048,
        num_blocks=num_blocks,
        r_num_blocks=0,
        dropout_rate=0.1,
        positional_dropout_rate=0.1,
        self_attention_dropout_rate=0.0,
        src_attention_dropout_rate=0.0,
        input_layer="embed",
        use_output_layer=True,
        normalize_before=True,
        src_attention=True,
        query_bias=True,
        key_bias=True,
        value_bias=True,
        activation_type="relu",
        gradient_checkpointing=False,
        tie_word_embedding=False,
        use_sdpa=False,
        layer_norm_type='layer_norm',
        norm_eps=1e-5,
        n_kv_head=None,
        head_dim=None,
        mlp_type='position_wise_feed_forward',
        mlp_bias=True,
        n_expert=8,
        n_expert_activated=2
    )

    # 4) Compare state_dicts
    model_sd = decoder.left_decoder.state_dict()
    ckpt_sd  = {
        k[len("decoder.left_decoder."):]: v
        for k, v in ckpt.items()
        if k.startswith("decoder.left_decoder.")
    }

    missing = []
    mismatched = []
    extra = sorted(set(ckpt_sd.keys()) - set(model_sd.keys()))

    for name, param in model_sd.items():
        if name not in ckpt_sd:
            missing.append(name)
        elif tuple(param.shape) != tuple(ckpt_sd[name].shape):
            mismatched.append((name, tuple(param.shape), tuple(ckpt_sd[name].shape)))

    # 5) Print results
    print("=== Verification Result ===")
    if not missing and not mismatched and not extra:
        print("✅ All parameters match between model and checkpoint.")
    else:
        if missing:
            print("\n❌ Missing in checkpoint:")
            for k in missing:
                print("   -", k)
        if mismatched:
            print("\n❌ Shape mismatches:")
            for k, mshape, cshape in mismatched:
                print(f"   - {k}: model={mshape}, checkpoint={cshape}")
        if extra:
            print("\n⚠️ Extra keys in checkpoint:")
            for k in extra:
                print("   -", k)

# if __name__ == "__main__":
#     import sys
#     if len(sys.argv) != 2:
#         print("Usage: python compare_decoder.py /path/to/chunkformer-checkpoint")
#         sys.exit(1)

path = "../../chunkformer-large-vie"  # adjust path
instantiate_and_compare(path)
