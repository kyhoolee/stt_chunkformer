import torch
import torchaudio.functional as F
import torchaudio.transforms as T
import random

class AudioAugmenter:
    _resampler_cache = {}
    _resampler_sr = None
    _resampler_initialized = False

    def __init__(self, sample_rate: int):
        self.sr = sample_rate
        self.ir_kernel = self._get_reverb_kernel()

        # Check and initialize static cache
        self._init_static_resamplers()

    def _init_static_resamplers(self):
        if AudioAugmenter._resampler_initialized:
            if AudioAugmenter._resampler_sr != self.sr:
                raise ValueError(
                    f"[AUGMENTER ERROR] Static resampler cache was initialized for sample_rate={AudioAugmenter._resampler_sr}, "
                    f"but got new instance with sample_rate={self.sr}."
                )
            return  # already initialized with correct SR

        # Otherwise, first time init
        AudioAugmenter._resampler_cache = {}
        AudioAugmenter._resampler_sr = self.sr
        AudioAugmenter._resampler_initialized = True

    def _get_resampler(self, new_sr):
        key = (self.sr, new_sr)
        if key not in AudioAugmenter._resampler_cache:
            print(f"   🛠️  [Resampler] Creating new resampler {key}")
            AudioAugmenter._resampler_cache[key] = T.Resample(orig_freq=self.sr, new_freq=new_sr)
        return AudioAugmenter._resampler_cache[key]

    def _get_reverse_resampler(self, new_sr):
        key = (new_sr, self.sr)
        if key not in AudioAugmenter._resampler_cache:
            print(f"   🛠️  [Resampler] Creating reverse resampler {key}")
            AudioAugmenter._resampler_cache[key] = T.Resample(orig_freq=new_sr, new_freq=self.sr)
        return AudioAugmenter._resampler_cache[key]

    def speed_perturb(self, wav):
        speed = random.uniform(0.9, 1.1)
        orig_len = wav.shape[1]
        new_sr = int(self.sr * speed)

        print(f"   ⚙️  [speed_perturb] speed={speed:.3f}, new_sr={new_sr}")
        print(f"      📥 input shape: {wav.shape}, orig_len: {orig_len}")

        try:
            up = self._get_resampler(new_sr)
            wav = up(wav)
            print(f"      🔁 Resample #1 → shape: {wav.shape}")
        except Exception as e:
            print(f"❌ [RESAMPLE 1 ERROR] - {e}")
            raise

        try:
            down = self._get_reverse_resampler(new_sr)
            wav = down(wav)
            print(f"      🔁 Resample #2 → shape: {wav.shape}")
        except Exception as e:
            print(f"❌ [RESAMPLE 2 ERROR] - {e}")
            raise

        out = wav[:, :orig_len]
        print(f"      ✅ Final shape after truncate: {out.shape}")
        return out

    def vol_perturb(self, wav):
        factor = 0.8 + (1.2 - 0.8) * torch.rand(1).item()
        return wav * factor

    def telephony_effect(self, wav):
        return F.bandpass_biquad(wav, self.sr, central_freq=1700.0, Q=0.707)

    def noise_mix(self, wav, snr_db=20):
        noise = torch.randn_like(wav)
        signal_power = wav.norm(p=2)
        noise_power = noise.norm(p=2)
        factor = (signal_power / noise_power) / (10 ** (snr_db / 20))
        return wav + factor * noise

    def pitch_shift(self, wav):
        ratio = random.uniform(0.95, 1.05)
        new_sr = int(self.sr * ratio)
        up = T.Resample(orig_freq=self.sr, new_freq=new_sr)
        down = T.Resample(orig_freq=new_sr, new_freq=self.sr)
        wav = down(up(wav))
        return wav[:, :wav.shape[1]]

    def _get_reverb_kernel(self):
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
