import torch
from torchaudio.compliance.kaldi import fbank
from jiwer import wer
from datasets import load_dataset, Audio
import torchaudio
import os
from ..decode import init, load_audio
from ..model.utils.ctc_utils import get_output, get_output_with_timestamps


class Args:
    model_checkpoint = "chunkformer-large-vie"
    total_batch_duration = 1800
    chunk_size = 64
    left_context_size = 128
    right_context_size = 128
    long_form_audio = None
    label_text = None


args = Args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def prepare_input(waveform_data):
    if isinstance(waveform_data, torch.Tensor):
        waveform = waveform_data
    else:
        waveform = torch.from_numpy(waveform_data).float()

    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)

    waveform = waveform.to(device)

    feats = fbank(
        waveform,
        num_mel_bins=80,
        frame_length=25,
        frame_shift=10,
        dither=0.0,
        energy_floor=0.0,
        sample_frequency=16000,
    ).unsqueeze(0).to(device)
    return feats


@torch.no_grad()
def decode_short_batch(xs, model, char_dict, args):
    chunk_size = args.chunk_size
    left_context_size = args.left_context_size
    right_context_size = args.right_context_size
    device = next(model.parameters()).device

    x_len = torch.tensor([xs.shape[1]], dtype=torch.int, device=device)
    offset = torch.zeros(1, dtype=torch.int, device=device)

    encoder_outs, encoder_lens, n_chunks, _, _, _ = model.encoder.forward_parallel_chunk(
        xs=[xs.squeeze(0)],
        xs_origin_lens=x_len,
        chunk_size=chunk_size,
        left_context_size=left_context_size,
        right_context_size=right_context_size,
        offset=offset,
    )

    hyps = model.encoder.ctc_forward(encoder_outs, encoder_lens, n_chunks)
    pred = get_output(hyps, char_dict)[0]
    return pred


@torch.no_grad()
def decode_long_form(xs, model, char_dict):
    def get_max_input_context(c, r, n):
        return r + max(c, r) * (n - 1)

    subsampling_factor = model.encoder.embed.subsampling_factor
    chunk_size = args.chunk_size
    left_context_size = args.left_context_size
    right_context_size = args.right_context_size
    conv_lorder = model.encoder.cnn_module_kernel // 2

    max_len = int(args.total_batch_duration // 0.01) // 2
    multiply_n = max_len // chunk_size // subsampling_factor
    truncated_context_size = chunk_size * multiply_n

    rel_right_context_size = get_max_input_context(
        chunk_size, max(right_context_size, conv_lorder), model.encoder.num_blocks
    ) * subsampling_factor

    offset = torch.zeros(1, dtype=torch.int, device=device)
    num_blocks = model.encoder.num_blocks
    att_cache = torch.zeros((
        num_blocks,
        left_context_size,
        model.encoder.attention_heads,
        model.encoder._output_size * 2 // model.encoder.attention_heads
    ), device=device)
    cnn_cache = torch.zeros((
        num_blocks,
        model.encoder._output_size,
        conv_lorder
    ), device=device)

    hyps = []
    for idx in range(0, xs.shape[1], truncated_context_size * subsampling_factor):
        start = max(truncated_context_size * subsampling_factor * idx, 0)
        end = min(truncated_context_size * subsampling_factor * (idx + 1) + 7, xs.shape[1])
        x = xs[:, start:end + rel_right_context_size]
        x_len = torch.tensor([x[0].shape[0]], dtype=torch.int, device=device)

        encoder_outs, encoder_lens, _, att_cache, cnn_cache, offset = model.encoder.forward_parallel_chunk(
            xs=x,
            xs_origin_lens=x_len,
            chunk_size=chunk_size,
            left_context_size=left_context_size,
            right_context_size=right_context_size,
            att_cache=att_cache,
            cnn_cache=cnn_cache,
            truncated_context_size=truncated_context_size,
            offset=offset
        )

        encoder_outs = encoder_outs.reshape(1, -1, encoder_outs.shape[-1])[:, :encoder_lens]
        if chunk_size * multiply_n * subsampling_factor * idx + rel_right_context_size < xs.shape[1]:
            encoder_outs = encoder_outs[:, :truncated_context_size]

        offset = offset - encoder_lens + encoder_outs.shape[1]

        hyp = model.encoder.ctc_forward(encoder_outs).squeeze(0)
        hyps.append(hyp)

        if chunk_size * multiply_n * subsampling_factor * idx + rel_right_context_size >= xs.shape[1]:
            break

    hyps = torch.cat(hyps)
    decode = get_output_with_timestamps([hyps], char_dict)[0]
    full_text = " ".join([item["decode"] for item in decode])
    return full_text


def main():
    ds = load_dataset("AILAB-VNUHCM/vivos", split="test", trust_remote_code=True)
    ds = ds.cast_column("audio", Audio(sampling_rate=16000)).with_format("torch")

    model, char_dict = init(args.model_checkpoint, device)
    model.eval()

    print("✅ Model and char_dict initialized.")
    print(f"🔤 Sample char_dict: {list(char_dict.keys())[:10]}")

    os.makedirs("debug_wavs", exist_ok=True)


    total_wer = 0
    count = 20
    for i in range(count):
        waveform = ds[i]["audio"]["array"]

        # Save waveform to WAV for debugging
        save_path = f"debug_wavs/sample_{i:02d}.wav"
        torchaudio.save(save_path, torch.tensor(waveform).unsqueeze(0), sample_rate=16000)
        print(f"[#{i:02d}] 💾 Saved to {save_path}")



        gt_text = ds[i]["sentence"]
        duration_sec = len(waveform) / 16000

        print(f"\n[#{i:02d}] 🎧 Audio duration: {duration_sec:.2f}s")

        xs = prepare_input(waveform)
        print(f"[#{i:02d}] 🎛️ fbank shape: {xs.shape} (Batch, Time, Mel)")

        if duration_sec <= 2.5:
            print(f"[#{i:02d}] 🚀 Using SHORT decoding (≤ 2.5s)")
            pred_text = decode_short_batch(xs, model, char_dict, args)
        else:
            print(f"[#{i:02d}] 🧠 Using LONG decoding (> 2.5s)")
            pred_text = decode_long_form(xs, model, char_dict)

        score = wer(gt_text, pred_text)
        total_wer += score

        print(f"[#{i:02d}] ✅ WER: {score:.2f}")
        print(f"[#{i:02d}] 🔵 Ground Truth:", gt_text)
        print(f"[#{i:02d}] 🟢 Prediction  :", pred_text)

    print(f"\n📊 Final Avg WER ({count} samples): {total_wer / count:.4f}")


if __name__ == "__main__":
    main()
