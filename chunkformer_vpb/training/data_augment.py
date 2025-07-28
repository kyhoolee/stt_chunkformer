# augment/audio_augmenter.py
import torch 
import torchaudio
import torchaudio.functional as F
import random

class AudioAugmenter:
    def __init__(self, sample_rate: int):
        self.sr = sample_rate

    def vol_perturb(self, wav):
        factor = 0.8 + (1.2 - 0.8) * torch.rand(1).item()
        return wav * factor

    def speed_perturb(self, wav):
        # speed = torch.choice(torch.tensor([0.9, 1.0, 1.1]))  # or random.uniform
        speed = random.uniform(0.9, 1.1)
        orig_len = wav.shape[1]
        new_sr = int(self.sr * speed)
        wav = torchaudio.transforms.Resample(orig_freq=self.sr, new_freq=new_sr)(wav)
        wav = torchaudio.transforms.Resample(orig_freq=new_sr, new_freq=self.sr)(wav)
        return wav[:, :orig_len]

    def telephony_effect(self, wav):
        return F.bandpass_biquad(wav, self.sr, central_freq=1700.0, Q=0.707)

    def apply(self, wav, mode: str):
        if mode == "vol":
            return self.vol_perturb(wav)
        elif mode == "speed":
            return self.speed_perturb(wav)
        elif mode == "telephony":
            return self.telephony_effect(wav)
        else:
            return wav
