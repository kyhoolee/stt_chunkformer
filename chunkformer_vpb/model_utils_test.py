import torch
from torchaudio.compliance.kaldi import fbank
from torch.utils.benchmark import Timer
from jiwer import wer
from datasets import load_dataset
from datasets import Audio

from .decode import init, load_audio
from .model.utils.ctc_utils import get_output_with_timestamps




class Args:
    model_checkpoint = "../chunkformer-large-vie"
    total_batch_duration = 1800
    chunk_size = 64
    left_context_size = 128
    right_context_size = 128
    long_form_audio = "../debug_wavs/sample_00.wav"
    label_text = "NỬA VÒNG TRÁI ĐẤT HƠN BẢY NĂM".lower()


    # long_form_audio = "stt_chunkformer/data/common_voice_vi_23397238.wav"
    # label_text = "thảo cố hết sức đạp vào người có em để thụ ra ngoài"
    # long_form_audio = "stt_chunkformer/data/10000.wav"
    # label_text = "thế là sáng hôm sau cái tin tôi hỏi thăm đường đã lan ra khắp xóm"
    # long_form_audio = "debug_wavs/sample_19.wav"
    # label_text = "CHẬM NẮM BẮT XU HƯỚNG PHÁT TRIỂN CÔNG NGHỆ MỚI".lower()
    



args = Args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

voice_model, voice_char_dict = init(args.model_checkpoint, device)
voice_model.eval()


def prepare_input_file(audio_path, device):
    waveform = load_audio(audio_path)
    print(f"  🔍 [DEBUG] Loaded waveform shape: {waveform.shape}, dtype: {waveform.dtype}")

    print("📊 waveform.min():", waveform.min().item())
    print("📊 waveform.max():", waveform.max().item())
    print("📊 waveform.mean():", waveform.mean().item())



    feats = fbank(
        waveform,
        num_mel_bins=80,
        frame_length=25,
        frame_shift=10,
        dither=0.0,
        energy_floor=0.0,
        sample_frequency=16000,
    ).unsqueeze(0).to(device)
    print(f"  🔍 [DEBUG] fbank shape: {feats.shape}")  # Expect something like (1, time_steps, 80)

    return feats

def prepare_input_waveform(waveform_data, device): # Modified to accept waveform data directly
    # waveform_data is expected to be a PyTorch tensor or numpy array
    # If it's a numpy array, convert it to a tensor
    print(f"  🔍 [DEBUG] prepare_input_waveform called with type: {type(waveform_data)}")
    print(f"  🔍 [DEBUG] waveform_data shape: {waveform_data.shape if hasattr(waveform_data, 'shape') else 'N/A'}")
    if isinstance(waveform_data, torch.Tensor):
        print("Input is a PyTorch tensor.")
        waveform = waveform_data
    else:
        print("Input is a numpy array, converting to tensor.")
        waveform = torch.from_numpy(waveform_data).float()

    # If the waveform is mono, ensure it has a channel dimension (batch_size, num_samples)
    if waveform.dim() == 1:
        print("Input waveform is 1D, adding channel dimension.")
        waveform = waveform.unsqueeze(0) # Add a batch dimension if it's just samples

    print(f"  🔍 [DEBUG] waveform shape after processing: {waveform.shape}, dtype: {waveform.dtype}")

    # Ensure waveform is on the correct device for fbank
    waveform = waveform.to(device)

    print("📊 waveform.min():", waveform.min().item())
    print("📊 waveform.max():", waveform.max().item())
    print("📊 waveform.mean():", waveform.mean().item())

    # Check if need to convert from [-1.0, 1.0] range to [-32768, 32767]
    if waveform.abs().max() <= 1.0:
        print("📦 Normalized waveform detected. Rescaling to int16-style range.")
        waveform = waveform * 32768.0

    # Ensure waveform is in the correct range
    waveform = waveform.clamp(-32768, 32767)

    # If your load_audio function from stt_chunkformer/decode.py handles a path
    # and returns a pydub.AudioSegment, you might need to adjust this.
    # However, since you're using `ds[i]["audio"]["array"]`, you already have the waveform.
    # You might not need load_audio directly from the file system.

    print("📊 waveform.min():", waveform.min().item())
    print("📊 waveform.max():", waveform.max().item())
    print("📊 waveform.mean():", waveform.mean().item())

    feats = fbank(
        waveform,
        num_mel_bins=80,
        frame_length=25,
        frame_shift=10,
        dither=0.0,
        energy_floor=0.0,
        sample_frequency=16000, # Ensure this matches the dataset's sample rate
    ).unsqueeze(0).to(device)
    print(f"  🔍 [DEBUG] fbank shape: {feats.shape}")  # Expect something like (1, time_steps, 80)
    return feats


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
    for idx, _ in enumerate(range(0, xs.shape[1], truncated_context_size * subsampling_factor)):
        
        
        start = max(truncated_context_size * subsampling_factor * idx, 0)
        end = min(truncated_context_size * subsampling_factor * (idx + 1) + 7, xs.shape[1])
        x = xs[:, start:end + rel_right_context_size]
        x_len = torch.tensor([x[0].shape[0]], dtype=torch.int, device=device)

        print(f"  🔍 [DEBUG] Chunk {idx} | Input x shape: {x.shape} | x_len: {x_len}")


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

        print(f"  🔍 [DEBUG] encoder_outs shape: {encoder_outs.shape}")


        encoder_outs = encoder_outs.reshape(1, -1, encoder_outs.shape[-1])[:, :encoder_lens]
        if chunk_size * multiply_n * subsampling_factor * idx + rel_right_context_size < xs.shape[1]:
            encoder_outs = encoder_outs[:, :truncated_context_size]

        offset = offset - encoder_lens + encoder_outs.shape[1]

        # ❌ bỏ cái này
        hyp = model.encoder.ctc_forward(encoder_outs).squeeze(0)
        hyps.append(hyp)

        # ✅ thay bằng:
        # logits = model.ctc.log_softmax(encoder_outs).squeeze(0)
        # hyps.append(logits)


        if chunk_size * multiply_n * subsampling_factor * idx + rel_right_context_size >= xs.shape[1]:
            break

    hyps = torch.cat(hyps)
    print(f"  🔍 [DEBUG] Full hyp shape: {hyps.shape}")
    print(f"  🔍 [DEBUG] Full hyp mean: {hyps}")
    decode = get_output_with_timestamps([hyps], char_dict)[0]
    print(f"  🔍 [DEBUG] decode output: {decode}")
    full_text = " ".join([item["decode"] for item in decode])

    print("====== DEBUG LAYER STATE ======")
    print("  ✅ device:", device)
    print("  ✅ CTC Linear weight mean:", model.ctc.ctc_lo.weight.data.mean().item())
    print("  ✅ Encoder out mean:", encoder_outs.mean().item())
    # print("  ✅ hyp mean:", hyp.mean().item())
    # print("  ✅ hyp argmax:", hyp.argmax(dim=-1))
    print("================================")


    return full_text
