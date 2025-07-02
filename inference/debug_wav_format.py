import torch
import torchaudio
from datasets import load_dataset, Audio
import os

def inspect_waveform(waveform, sample_rate, label=None, tag="Original"):
    print(f"\n🔍 [{tag}] waveform info:")
    print(f"📐 Shape       : {waveform.shape}")
    print(f"🔢 Dtype       : {waveform.dtype}")
    print(f"🎵 Sample rate : {sample_rate}")
    print(f"📊 Min         : {waveform.min().item():.4f}")
    print(f"📊 Max         : {waveform.max().item():.4f}")
    print(f"📊 Mean        : {waveform.mean().item():.4f}")
    if label:
        print(f"🗣️  Transcript  : {label}")

def save_waveform(waveform, save_path, sample_rate=16000):
    # if waveform.abs().max() <= 1.0:
    #     print("📦 Normalized waveform detected. Rescaling to int16-style range.")
    #     waveform = waveform * 32768.0
    # waveform = waveform.clamp(-32768, 32767).short()
    torchaudio.save(save_path, waveform.float(), sample_rate, encoding="PCM_S", bits_per_sample=16)
    print(f"💾 Saved waveform to {save_path}")

def main():
    ds = load_dataset("AILAB-VNUHCM/vivos", split="test", trust_remote_code=True)
    ds = ds.cast_column("audio", Audio(sampling_rate=16000))
    ds = ds.with_format("torch")

    i = 0  # test sample index
    audio = ds[i]["audio"]
    waveform = audio["array"]
    sample_rate = audio["sampling_rate"]
    label = ds[i]["sentence"]

    inspect_waveform(waveform, sample_rate, label=label, tag="Original from datasets")

    # Save as PCM WAV
    folder = "debug_wavs_1"
    os.makedirs(folder, exist_ok=True)
    save_path = f"{folder}/sample_{i:02d}.wav"
    save_waveform(torch.tensor(waveform).unsqueeze(0), save_path, sample_rate)

    # Load back
    loaded_waveform, sr = torchaudio.load(save_path)
    inspect_waveform(loaded_waveform, sr, tag="Loaded from .wav")

    # Optional: Compare waveform shape or mean diff
    diff = (loaded_waveform - waveform.unsqueeze(0)).abs().mean().item()
    print(f"📏 Mean abs diff between original and reloaded: {diff:.4f}")

if __name__ == "__main__":
    main()
