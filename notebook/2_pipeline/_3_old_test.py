# %%
import torch
from torchaudio.compliance.kaldi import fbank
from torch.utils.benchmark import Timer
from jiwer import wer
import os
from chunkformer_vpb.model_utils import prepare_input_file, decode_long_form, get_default_args
from chunkformer_vpb.model_utils import init, dump_module_structure, decode_aed_long_form



model_checkpoint = "../../chunkformer-large-vie"  # adjust if needed
output_dir = "model_architect"
os.makedirs(output_dir, exist_ok=True)

# === Load model ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model, char_dict = init(model_checkpoint, device)
model.eval()

args = get_default_args()
args.audio_path = "../../debug_wavs/sample_00.wav"
args.label_text = "nửa vòng trái đất hơn bảy năm"
feats = prepare_input_file(args.audio_path, device)
decoded = decode_long_form(feats, model, char_dict, args, device)

ctc_text = decoded
aed_text_raw, aed_text_clean = decode_aed_long_form(feats, model, char_dict, args, device)

print(f"🆚 So sánh:\n- CTC: {ctc_text}\n- AED: {aed_text_clean}")


print("🟢 Prediction     :", decoded)
if args.label_text:
    print("🔵 Ground Truth   :", args.label_text)
    print("❌ WER            :", wer(args.label_text.lower(), decoded.lower()))



#   --model_checkpoint ../chunkformer-large-vie \
#   --audio_path ../debug_wavs/sample_00.wav \
#   --label_text "nửa vòng trái đất hơn bảy năm"


# %%
import torch
from chunkformer_vpb.finetune_utils import (
    get_default_args,
    prepare_input_file,
    load_model_only,
    GreedyTokenizer,
    compute_chunkformer_loss,
)

def main():
    # 1. Chuẩn bị args và device
    args = get_default_args()
    args.model_checkpoint = "../../chunkformer-large-vie"     # đường dẫn đến folder checkpoint
    args.audio_path       = "../../debug_wavs/sample_00.wav"  # file audio mẫu
    args.label_text       = "nửa vòng trái đất hơn bảy năm"  # label ground-truth

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2. Load model + tokenizer
    model, _ = load_model_only(args.model_checkpoint, device)
    model.ctc_weight = 0.3
    # model.reverse_weight = 0.3 -> can not work due to there is no right_decoder 
    tokenizer = GreedyTokenizer(vocab_path=f"{args.model_checkpoint}/vocab.txt")

    # 3. Prepare input features
    xs = prepare_input_file(args.audio_path, device)  # [1, T_raw, 80]

    # 4. Compute loss
    loss_dict = compute_chunkformer_loss(
        model=model,
        tokenizer=tokenizer,
        xs=xs,
        args=args,
        label_text=args.label_text,
        device=device
    )

    # 5. In kết quả
    print(f"Total Loss: {loss_dict['loss'].item():.4f}")
    print(f"  CTC Loss: {loss_dict['loss_ctc'].item():.4f}")
    print(f"  AED Loss: {loss_dict['loss_att'].item():.4f}")

main()


# %%


# %%



