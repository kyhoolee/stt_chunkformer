# augment/audio_augmenter.py
import torch 
import torchaudio
import torchaudio.functional as F
import random

'''
⏱️ So sánh tốc độ giữa các loại augmentation
| Augment type             | Thời gian (tương đối)            | Nguyên nhân chính             |
| ------------------------ | -------------------------------- | ----------------------------- |
| `vol` (gain)             | ⚡ Rất nhanh (\~1ms)              | Chỉ là nhân hệ số             |
| `telephony` (bandpass)   | ⚡ Nhanh (\~1–2ms)                | Chỉ dùng `biquad filter`      |
| `speed`                  | 🐢 Trung bình (\~10–20ms/sample) | Gồm 2 lần `resample`          |
| `reverb`, `room`         | 🐌 Chậm hơn (\~50ms++)           | Convolve với impulse response |
| `noise mixing` (wav-add) | ⚡ Nhanh – nếu preload noise      | Cộng vector đơn giản          |

✅ Cách tối ưu hiệu quả
| Chiến lược                               | Mô tả                                        | Áp dụng                      |
| ---------------------------------------- | -------------------------------------------- | ---------------------------- |
| **Precompute augment offline**           | Lưu `.wav` augment ra file → load nhanh      | Khi tập cố định              |
| **Cache in-memory** (mini batch)         | Dùng `lru_cache` hoặc `dataset-level buffer` | Nếu RAM đủ                   |
| **`num_workers > 0`** trong `DataLoader` | Tăng song song `__getitem__`                 | Bắt buộc nếu dùng on-the-fly |
| **Chỉ augment một phần epoch**           | VD: 50% sample augment mỗi epoch             | Giảm tải thời gian           |


'''

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
