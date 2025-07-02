import torch
from torchaudio.compliance.kaldi import fbank
from torch.utils.benchmark import Timer
from jiwer import wer
from datasets import load_dataset
from datasets import Audio

from ..decode import init, load_audio
from ..model_utils import args, device, prepare_input_waveform, decode_long_form
from ..model_utils import voice_char_dict, voice_model


def main():
    # Load dataset
    ds = load_dataset("AILAB-VNUHCM/vivos", split="test", trust_remote_code=True)
    # Ensure the audio column is cast to the correct sampling rate if necessary.
    # The default for VIVOS is likely 16kHz, but it's good practice to specify.
    ds = ds.cast_column("audio", Audio(sampling_rate=16000))
    ds = ds.with_format("torch")

    # model, char_dict = init(args.model_checkpoint, device)
    # model.eval()
    model = voice_model
    char_dict = voice_char_dict
    model.eval()

    print(f"  📚 [DEBUG] char_dict keys: {list(char_dict.keys())[:20]}")


    total_wer = 0
    N = 1
    for i in range(N):  # you can increase this
        # Access the audio array directly from the dataset
        audio_array = ds[i]["audio"]["array"]
        label_text = ds[i]["sentence"]

        # Pass the audio array (waveform data) directly to prepare_input
        xs = prepare_input_waveform(audio_array, device)
        print(f"  🔍 [DEBUG] fbank shape: {xs.shape}")  # Expect something like (1, time_steps, 80)

        pred_text = decode_long_form(xs, model, char_dict)
        score = wer(label_text, pred_text)
        total_wer += score

        print(f"[#{i:02d}] WER: {score:.2f}")
        print("  🔵 GT:", label_text)
        print("  🟢 PR:", pred_text)
        print()

    avg_wer = total_wer / N
    print(f"📊 Avg WER ({N} samples): {avg_wer:.4f}")

if __name__ == "__main__":
    main()