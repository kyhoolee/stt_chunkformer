import time
import torchaudio
import torchaudio.functional as F
from pathlib import Path
from chunkformer_vpb.training.data_augment import AudioAugmenter

AUG_TYPES = ['vol', 'speed', 'telephony', 'noise', 'pitch', 'reverb']

SAMPLE_RATE = 16000
AUDIO_FOLDER = Path("../../../vpb_dataset/archive/tts_dataset_best_call_agent_audio/tts_dataset_best_call_agent_audio/wavs")

# Lấy N file mẫu
N_SAMPLE = 20
audio_files = sorted(list(AUDIO_FOLDER.glob("*.wav")))[:N_SAMPLE]

augmenter = AudioAugmenter(SAMPLE_RATE)

stats = {aug: [] for aug in AUG_TYPES}
durations = []  # tổng duration mỗi file

print(f"🔬 Benchmark {len(audio_files)} samples with sr={SAMPLE_RATE}")

for audio_file in audio_files:
    wav, sr = torchaudio.load(audio_file)
    duration = wav.shape[1] / SAMPLE_RATE
    durations.append(duration)

    if sr != SAMPLE_RATE:
        wav = F.resample(wav, orig_freq=sr, new_freq=SAMPLE_RATE)

    print(f"\n🎧 File: {audio_file.name} | Duration: {duration:.2f} sec")

    for aug in AUG_TYPES:
        t0 = time.perf_counter()
        _ = augmenter.apply(wav.clone(), aug)
        t1 = time.perf_counter()
        delta = t1 - t0
        stats[aug].append((delta, duration))  # lưu cả thời gian và độ dài audio
        print(f"   🧪 {aug:<10} → {delta*1000:.2f} ms")

# === Tổng hợp kết quả ===
print("\n📊 Tổng hợp:")
for aug in AUG_TYPES:
    records = stats[aug]
    if not records:
        continue

    total_time = sum(d for d, _ in records)
    total_audio_sec = sum(t for _, t in records)
    time_per_sec_audio = (total_time / total_audio_sec) * 1000  # ms trên mỗi giây audio

    print(f"🔹 {aug:<10}:")
    print(f"   • Tổng thời gian xử lý  : {total_time:.2f} sec")
    print(f"   • Tổng độ dài audio     : {total_audio_sec:.2f} sec")
    print(f"   • T/g mỗi 1s audio      : {time_per_sec_audio:.2f} ms/sec")
    print(f"   • Trung bình mỗi file   : {total_time / len(records)*1000:.2f} ms")
