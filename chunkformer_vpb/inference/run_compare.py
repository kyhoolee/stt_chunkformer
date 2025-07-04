import torch
from torchaudio.compliance.kaldi import fbank
from torch.utils.benchmark import Timer
from jiwer import wer

from ..model_utils_test import args, device, prepare_input_file, prepare_input_waveform, decode_long_form
from ..model_utils_test import voice_char_dict, voice_model




# === Load model
# model, char_dict = init(args.model_checkpoint, device)
model = voice_model
char_dict = voice_char_dict
model.eval()



print(f"  📚 [DEBUG] char_dict keys: {list(char_dict.keys())[:20]}")


# === Prepare input
xs = prepare_input_file(args.long_form_audio, device)

# === Run + Benchmark
def decode_and_return():
    return decode_long_form(xs, model, char_dict)


t = Timer(
    stmt="decode_and_return()",
    setup="from __main__ import decode_and_return",
    globals=globals()
)

pred_text = decode_and_return()
time_ms = t.timeit(1).mean * 1000

print("⏱ Time (1 sample): {:.2f}ms".format(time_ms))
print("🟢 Prediction     :", pred_text)
print("🔵 Ground Truth   :", args.label_text)
print("❌ WER            :", wer(args.label_text, pred_text))
