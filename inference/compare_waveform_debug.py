import torch
import torchaudio
import numpy as np
from datasets import load_dataset, Audio
from torchaudio.compliance.kaldi import fbank
import matplotlib.pyplot as plt
import os


def compare_waveforms(idx=0, wav_folder="debug_wavs"):
    # Load dataset (VIVOS test set)
    ds = load_dataset("AILAB-VNUHCM/vivos", split="test", trust_remote_code=True)
    ds = ds.cast_column("audio", Audio(sampling_rate=16000)).with_format("torch")

    # === Load waveform from dataset
    ds_waveform = ds[idx]["audio"]["array"]
    ds_waveform = torch.from_numpy(ds_waveform).float() if isinstance(ds_waveform, np.ndarray) else ds_waveform
    if ds_waveform.dim() == 1:
        ds_waveform = ds_waveform.unsqueeze(0)

    # === Load waveform from saved .wav file
    wav_path = os.path.join(wav_folder, f"sample_{idx:02d}.wav")
    file_waveform, sr = torchaudio.load(wav_path)

    assert sr == 16000, "Sample rate mismatch"

    # === Print basic info
    print(f"🧪 Comparing sample #{idx}")
    print(f"  Dataset waveform shape: {ds_waveform.shape}, dtype: {ds_waveform.dtype}, max: {ds_waveform.max():.4f}")
    print(f"  File waveform    shape: {file_waveform.shape}, dtype: {file_waveform.dtype}, max: {file_waveform.max():.4f}")

    # === Check waveform closeness
    if ds_waveform.shape != file_waveform.shape:
        min_len = min(ds_waveform.shape[-1], file_waveform.shape[-1])
        ds_waveform = ds_waveform[:, :min_len]
        file_waveform = file_waveform[:, :min_len]

    diff = (ds_waveform - file_waveform).abs()
    print(f"  🔍 Max difference in waveform: {diff.max().item():.6f}")
    print(f"  🔍 Mean difference in waveform: {diff.mean().item():.6f}")
    print(f"  ✅ torch.allclose: {torch.allclose(ds_waveform, file_waveform, atol=1e-4)}")

    # === Compare FBANK features
    ds_fbank = fbank(ds_waveform, num_mel_bins=80, frame_length=25, frame_shift=10,
                     dither=0.0, energy_floor=0.0, sample_frequency=16000)
    file_fbank = fbank(file_waveform, num_mel_bins=80, frame_length=25, frame_shift=10,
                       dither=0.0, energy_floor=0.0, sample_frequency=16000)

    print(f"  📊 FBANK shape (dataset): {ds_fbank.shape}")
    print(f"  📊 FBANK shape (file):    {file_fbank.shape}")

    min_len = min(ds_fbank.shape[0], file_fbank.shape[0])
    fb_diff = (ds_fbank[:min_len] - file_fbank[:min_len]).abs()
    print(f"  🔍 Max FBANK diff: {fb_diff.max().item():.6f}")
    print(f"  🔍 Mean FBANK diff: {fb_diff.mean().item():.6f}")
    print(f"  ✅ torch.allclose (FBANK): {torch.allclose(ds_fbank[:min_len], file_fbank[:min_len], atol=1e-3)}")

    # === Plot overlay waveform (first 1000 samples)
    plt.figure(figsize=(12, 4))
    plt.plot(ds_waveform[0, :1000].numpy(), label="dataset")
    plt.plot(file_waveform[0, :1000].numpy(), label="file", alpha=0.7)
    plt.title(f"Waveform overlay - sample {idx}")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    compare_waveforms(idx=0)
