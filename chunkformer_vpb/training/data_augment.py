import torch 
import torchaudio
import torchaudio.functional as F
import random

class AudioAugmenter:
    def __init__(self, sample_rate: int):
        self.sr = sample_rate
        self.ir_kernel = self._get_reverb_kernel()

    def vol_perturb(self, wav):
        factor = 0.8 + (1.2 - 0.8) * torch.rand(1).item()
        return wav * factor

    def speed_perturb(self, wav):
        import time

        speed = random.uniform(0.9, 1.1)
        orig_len = wav.shape[1]
        new_sr = int(self.sr * speed)

        # print(f"   ⚙️  [speed_perturb] speed={speed:.3f}, new_sr={new_sr}")
        # print(f"      📥 input shape: {wav.shape}, orig_len: {orig_len}")

        t1 = time.time()
        try:
            wav = F.resample(wav, orig_freq=self.sr, new_freq=new_sr)
            # print(f"      🔁 Resample #1 → shape: {wav.shape} [{round(time.time() - t1, 2)}s]")
        except Exception as e:
            print(f"❌ [RESAMPLE 1 ERROR] - {e}")
            raise

        t2 = time.time()
        try:
            wav = F.resample(wav, orig_freq=new_sr, new_freq=self.sr)
            # print(f"      🔁 Resample #2 → shape: {wav.shape} [{round(time.time() - t2, 2)}s]")
        except Exception as e:
            print(f"❌ [RESAMPLE 2 ERROR] - {e}")
            raise

        out = wav[:, :orig_len]
        # print(f"      ✅ Final shape after truncate: {out.shape}")
        return out


    def telephony_effect(self, wav):
        return F.bandpass_biquad(wav, self.sr, central_freq=1700.0, Q=0.707)

    def noise_mix(self, wav, snr_db=20):
        """Add Gaussian noise to the waveform"""
        noise = torch.randn_like(wav)
        signal_power = wav.norm(p=2)
        noise_power = noise.norm(p=2)
        factor = (signal_power / noise_power) / (10 ** (snr_db / 20))
        return wav + factor * noise

    def pitch_shift(self, wav):
        """Pitch shift đơn giản bằng cách resample theo tỉ lệ tần số"""
        ratio = random.uniform(0.95, 1.05)
        new_sr = int(self.sr * ratio)
        wav = F.resample(wav, self.sr, new_sr)
        wav = F.resample(wav, new_sr, self.sr)
        return wav[:, :wav.shape[1]]

    def _get_reverb_kernel(self):
        """Create simple impulse response (decaying exponential)"""
        decay = torch.exp(-torch.linspace(0, 3, int(0.3 * self.sr)))
        kernel = decay.unsqueeze(0).unsqueeze(0)  # [1,1,T]
        return kernel

    def reverb_simple(self, wav):
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)
        return torch.nn.functional.conv1d(wav.unsqueeze(0), self.ir_kernel, padding="same").squeeze(0)

    def apply(self, wav, mode: str):
        if mode == "vol":
            return self.vol_perturb(wav)
        elif mode == "speed":
            return self.speed_perturb(wav)
        elif mode == "telephony":
            return self.telephony_effect(wav)
        elif mode == "noise":
            return self.noise_mix(wav)
        elif mode == "pitch":
            return self.pitch_shift(wav)
        elif mode == "reverb":
            return self.reverb_simple(wav)
        else:
            return wav
