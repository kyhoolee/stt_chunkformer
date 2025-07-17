import os
import torch
import torchaudio
import yaml
import jiwer
import argparse
import pandas as pd

from tqdm import tqdm
from colorama import Fore, Style

import torchaudio.compliance.kaldi as kaldi
from .model.utils.init_model import init_model
from .model.utils.checkpoint import load_checkpoint
from .model.utils.file_utils import read_symbol_table
from .model.utils.ctc_utils import get_output_with_timestamps, get_output
from contextlib import nullcontext
from pydub import AudioSegment
import torch
import torchaudio
import os
import torch
import yaml
from wenet.transformer.decoder import BiTransformerDecoder

@torch.no_grad()
def init(model_checkpoint, device):
    config_path = os.path.join(model_checkpoint, "config.yaml")
    checkpoint_path = os.path.join(model_checkpoint, "pytorch_model.bin")
    symbol_table_path = os.path.join(model_checkpoint, "vocab.txt")

    # 1. Load config
    with open(config_path, 'r') as fin:
        config = yaml.load(fin, Loader=yaml.FullLoader)

    # 2. Init model (encoder + ctc)
    model = init_model(config, config_path)
    model.eval()

    # 3. Load full checkpoint
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    load_checkpoint(model, checkpoint_path)

    # 4. Add decoder (AED) if exists in checkpoint
    if any(k.startswith("decoder.left_decoder") for k in ckpt):
        vocab_size = ckpt["decoder.left_decoder.embed.0.weight"].size(0)
        print(f"!!!Vocab size: {vocab_size}")
        d_model = ckpt["decoder.left_decoder.after_norm.weight"].size(0)
        block_ids = {int(k.split('.')[3]) for k in ckpt if k.startswith("decoder.left_decoder.decoders.")}
        num_blocks = max(block_ids) + 1

        decoder = BiTransformerDecoder(
            vocab_size=vocab_size,
            encoder_output_size=d_model,
            attention_heads=8,
            linear_units=2048,
            num_blocks=num_blocks,
            r_num_blocks=0,
            dropout_rate=0.1,
            positional_dropout_rate=0.1,
            self_attention_dropout_rate=0.0,
            src_attention_dropout_rate=0.0,
            input_layer="embed",
            use_output_layer=True,
            normalize_before=True,
            src_attention=True,
            query_bias=True,
            key_bias=True,
            value_bias=True,
            activation_type="relu",
            gradient_checkpointing=False,
            tie_word_embedding=False,
            use_sdpa=False,
            layer_norm_type='layer_norm',
            norm_eps=1e-5,
            n_kv_head=None,
            head_dim=None,
            mlp_type='position_wise_feed_forward',
            mlp_bias=True,
            n_expert=8,
            n_expert_activated=2
        )

        # ✅ Load đúng phần left_decoder
        decoder.left_decoder.load_state_dict({
            k[len("decoder.left_decoder."):]: v
            for k, v in ckpt.items()
            if k.startswith("decoder.left_decoder.")
        }, strict=False)

        decoder = decoder.to(device)
        decoder.eval()

        model.decoder = decoder  # Attach to main model

    # 5. Move parts to device
    model.encoder = model.encoder.to(device)
    model.ctc = model.ctc.to(device)

    # 6. Build char dictionary
    symbol_table = read_symbol_table(symbol_table_path)
    char_dict = {v: k for k, v in symbol_table.items()}

    return model, char_dict


# def load_audio(audio_path):
#     audio = AudioSegment.from_file(audio_path)
#     audio = audio.set_frame_rate(16000)
#     audio = audio.set_sample_width(2)  # set bit depth to 16bit
#     audio = audio.set_channels(1)  # set to mono
#     audio = torch.as_tensor(audio.get_array_of_samples(), dtype=torch.float32).unsqueeze(0)
#     return audio



def load_audio(audio_path):
    print(f"\n📥 Loading file: {audio_path}")

    # === 1. Dùng pydub để decode
    audio = AudioSegment.from_file(audio_path)
    print(f"🔍 [pydub] Raw frame_rate   : {audio.frame_rate}")
    print(f"🔍 [pydub] Sample width     : {audio.sample_width} bytes ({audio.sample_width * 8} bits)")
    print(f"🔍 [pydub] Channels         : {audio.channels}")
    print(f"🔍 [pydub] Duration (ms)    : {len(audio)} ms")

    # === 2. Chuẩn hóa về định dạng chuẩn (mono, 16kHz, 16-bit PCM)
    audio = audio.set_frame_rate(16000).set_sample_width(2).set_channels(1)

    # === 3. Trích waveform
    raw_array = audio.get_array_of_samples()
    print(f"🧪 [pydub] Type of array     : {type(raw_array)}, dtype: int{audio.sample_width * 8}")
    print(f"🧪 [pydub] First 10 samples  : {raw_array[:10]}")

    waveform = torch.as_tensor(raw_array, dtype=torch.float32).unsqueeze(0)  # Shape: (1, N)
    print(f"✅ [pydub] Waveform shape    : {waveform.shape}")
    print(f"📊 [pydub] Min: {waveform.min().item():.8f}, Max: {waveform.max().item():.8f}, Mean: {waveform.mean().item():.8f}")

    # === 4. So sánh với torchaudio (nếu cần)
    print("\n🔁 [Compare] Loading with torchaudio.load()")
    waveform_torch, sr_torch = torchaudio.load(audio_path)
    print(f"✅ [torchaudio] shape        : {waveform_torch.shape}, sample_rate: {sr_torch}")
    print(f"📊 [torchaudio] Min: {waveform_torch.min().item():.4f}, Max: {waveform_torch.max().item():.4f}, Mean: {waveform_torch.mean().item():.4f}")
    
    wav = waveform_torch
    if wav.dim() == 1:
        wav = wav.unsqueeze(0)
    if wav.abs().max() <= 1.0:
        wav = wav * 32768.0
    wav = wav.clamp(-32768, 32767)
    # compare normalization wav with pydub
    print(f"✅ [normalize] Normalized waveform shape: {wav.shape}")
    print(f"📊 [normalize] Normalized Min: {wav.min().item():.8f}, Max: {wav.max().item():.8f}, Mean: {wav.mean().item():.8f}")

    return waveform


@torch.no_grad()
def endless_decode(args, model, char_dict):    
    def get_max_input_context(c, r, n):
        return r + max(c, r) * (n-1)
    
    device = next(model.parameters()).device
    audio_path = args.long_form_audio
    # model configuration
    subsampling_factor = model.encoder.embed.subsampling_factor
    chunk_size = args.chunk_size
    left_context_size = args.left_context_size
    right_context_size = args.right_context_size
    conv_lorder = model.encoder.cnn_module_kernel // 2

    # get the maximum length that the gpu can consume
    max_length_limited_context = args.total_batch_duration
    max_length_limited_context = int((max_length_limited_context // 0.01))//2 # in 10ms second

    multiply_n = max_length_limited_context // chunk_size // subsampling_factor
    truncated_context_size = chunk_size * multiply_n # we only keep this part for text decoding

    # get the relative right context size
    rel_right_context_size = get_max_input_context(chunk_size, max(right_context_size, conv_lorder), model.encoder.num_blocks)
    rel_right_context_size = rel_right_context_size * subsampling_factor


    waveform = load_audio(audio_path)
    offset = torch.zeros(1, dtype=torch.int, device=device)

    # waveform = padding(waveform, sample_rate)
    xs = kaldi.fbank(waveform,
                            num_mel_bins=80,
                            frame_length=25,
                            frame_shift=10,
                            dither=0.0,
                            energy_floor=0.0,
                            sample_frequency=16000).unsqueeze(0)

    hyps = []
    att_cache = torch.zeros((model.encoder.num_blocks, left_context_size, model.encoder.attention_heads, model.encoder._output_size * 2 // model.encoder.attention_heads)).to(device)
    cnn_cache = torch.zeros((model.encoder.num_blocks, model.encoder._output_size, conv_lorder)).to(device)    # print(context_size)
    for idx, _ in tqdm(list(enumerate(range(0, xs.shape[1], truncated_context_size * subsampling_factor)))):
        start = max(truncated_context_size * subsampling_factor * idx, 0)
        end = min(truncated_context_size * subsampling_factor * (idx+1) + 7, xs.shape[1])

        x = xs[:, start:end+rel_right_context_size]
        x_len = torch.tensor([x[0].shape[0]], dtype=torch.int).to(device)

        encoder_outs, encoder_lens, _, att_cache, cnn_cache, offset = model.encoder.forward_parallel_chunk(xs=x, 
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
            encoder_outs = encoder_outs[:, :truncated_context_size]  # (B, maxlen, vocab_size) # exclude the output of rel right context
        offset = offset - encoder_lens + encoder_outs.shape[1]


        hyp = model.encoder.ctc_forward(encoder_outs).squeeze(0)
        hyps.append(hyp)
        if device.type == "cuda":
            torch.cuda.empty_cache()
        if chunk_size * multiply_n * subsampling_factor * idx + rel_right_context_size >= xs.shape[1]:
            break
    hyps = torch.cat(hyps)
    decode = get_output_with_timestamps([hyps], char_dict)[0]

    for item in decode:
        start = f"{Fore.RED}{item['start']}{Style.RESET_ALL}"
        end = f"{Fore.RED}{item['end']}{Style.RESET_ALL}"
        print(f"{start} - {end}: {item['decode']}")


@torch.no_grad()
def batch_decode(args, model, char_dict):
    df = pd.read_csv(args.audio_list, sep="\t")

    max_length_limited_context = args.total_batch_duration
    max_length_limited_context = int((max_length_limited_context // 0.01)) // 2 # in 10ms second    xs = []
    max_frames = max_length_limited_context
    chunk_size = args.chunk_size
    left_context_size = args.left_context_size
    right_context_size = args.right_context_size
    device = next(model.parameters()).device

    decodes = []
    xs = []
    xs_origin_lens = []
    for idx, audio_path in tqdm(enumerate(df['wav'].to_list())):
        print(f"Chunk {idx}: input {x.shape[1]} frames → encoder out {encoder_outs.shape[1]}")

        waveform = load_audio(audio_path)
        x = kaldi.fbank(waveform,
                                num_mel_bins=80,
                                frame_length=25,
                                frame_shift=10,
                                dither=0.0,
                                energy_floor=0.0,
                                sample_frequency=16000)

        xs.append(x)
        xs_origin_lens.append(x.shape[0])
        max_frames -= xs_origin_lens[-1]

        if (max_frames <= 0) or (idx == len(df) - 1):
            xs_origin_lens = torch.tensor(xs_origin_lens, dtype=torch.int, device=device)
            offset = torch.zeros(len(xs), dtype=torch.int, device=device)
            encoder_outs, encoder_lens, n_chunks, _, _, _ = model.encoder.forward_parallel_chunk(xs=xs, 
                                                                        xs_origin_lens=xs_origin_lens, 
                                                                        chunk_size=chunk_size,
                                                                        left_context_size=left_context_size,
                                                                        right_context_size=right_context_size,
                                                                        offset=offset
            )

            hyps = model.encoder.ctc_forward(encoder_outs, encoder_lens, n_chunks)
            print(f"Full hyp shape: {hyps.shape}")

            decodes += get_output(hyps, char_dict)
            print(f"Decodes len: ", len(decodes))
                                         

            # reset
            xs = []
            xs_origin_lens = []
            max_frames = max_length_limited_context


    df['decode'] = decodes
    if "txt" in df:
        wer = jiwer.wer(df["txt"].to_list(), decodes)
        print("WER: ", wer)
    df.to_csv(args.audio_list, sep="\t", index=False)



def main():
    # Create argument parser
    parser = argparse.ArgumentParser(description="Process arguments with default values.")

    # Add arguments with default values
    parser.add_argument(
        "--model_checkpoint", 
        type=str, 
        default=None, 
        help="Path to Huggingface checkpoint repo"
    )
    parser.add_argument(
        "--total_batch_duration", 
        type=int, 
        default=1800, 
        help="The total audio duration (in second) in a batch that your GPU memory can handle at once. Default is 1800s"
    )
    parser.add_argument(
        "--chunk_size", 
        type=int, 
        default=64, 
        help="Size of the chunks (default: 64)"
    )
    parser.add_argument(
        "--left_context_size", 
        type=int, 
        default=128, 
        help="Size of the left context (default: 128)"
    )
    parser.add_argument(
        "--right_context_size", 
        type=int, 
        default=128, 
        help="Size of the right context (default: 128)"
    )
    parser.add_argument(
        "--long_form_audio", 
        type=str, 
        default=None, 
        help="Path to the long audio file (default: None)"
    )
    parser.add_argument(
        "--audio_list", 
        type=str, 
        default=None, 
        required=False, 
        help="Path to the TSV file containing the audio list. The TSV file must have one column named 'wav'. If 'txt' column is provided, Word Error Rate (WER) is computed"
    )
    parser.add_argument(
        "--full_attn", 
        action="store_true",
        help="Whether to use full attention with caching. If not provided, limited-chunk attention will be used (default: False)"
    )
    parser.add_argument(
        "--device",
        type=torch.device,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run the model on (default: cuda if available else cpu)"
    )
    parser.add_argument(
        "--autocast_dtype",
        type=str,
        choices=["fp32", "bf16", "fp16"],
        default=None,
        help="Dtype for autocast. If not provided, autocast is disabled by default."
    )

    # Parse arguments
    args = parser.parse_args()
    device = torch.device(args.device)
    dtype = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16, None: None}[args.autocast_dtype]

    # Print the arguments
    print(f"Model Checkpoint: {args.model_checkpoint}")
    print(f"Device: {device}")
    print(f"Total Duration in a Batch (in second): {args.total_batch_duration}")
    print(f"Chunk Size: {args.chunk_size}")
    print(f"Left Context Size: {args.left_context_size}")
    print(f"Right Context Size: {args.right_context_size}")
    print(f"Long Form Audio Path: {args.long_form_audio}")
    print(f"Audio List Path: {args.audio_list}")
    
    assert args.model_checkpoint is not None, "You must specify the path to the model"
    assert args.long_form_audio or args.audio_list, "`long_form_audio` or `audio_list` must be activated"

    model, char_dict = init(args.model_checkpoint, device)
    with torch.autocast(device.type, dtype) if dtype is not None else nullcontext():
        if args.long_form_audio:
            print(">>> Long_form_audio:: ", args.long_form_audio)
            endless_decode(args, model, char_dict)
        else:
            print("<<< short form audio:: ")
            batch_decode(args, model, char_dict)

if __name__ == "__main__":
    main()

