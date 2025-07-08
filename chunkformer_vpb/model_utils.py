# chunkformer_vpb/inference/main_infer.py

import torch
import argparse
from torchaudio.compliance.kaldi import fbank
from jiwer import wer
from chunkformer_vpb.decode import init, load_audio
from chunkformer_vpb.model.utils.ctc_utils import get_output_with_timestamps
from chunkformer_vpb.model.utils.mask import make_pad_mask
# ==================== ARG PARSE ====================
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_checkpoint', type=str, required=True,
                        help='Path to model checkpoint folder (e.g. chunkformer-large-vie)')
    parser.add_argument('--audio_path', type=str, required=True,
                        help='Path to .wav audio file to transcribe')
    parser.add_argument('--label_text', type=str, default=None,
                        help='(Optional) Ground truth text for WER comparison')
    parser.add_argument('--chunk_size', type=int, default=64)
    parser.add_argument('--left_context_size', type=int, default=128)
    parser.add_argument('--right_context_size', type=int, default=128)
    parser.add_argument('--total_batch_duration', type=int, default=1800)  # in ms
    args = parser.parse_args()
    return args

def get_default_args():
    return argparse.Namespace(
        model_checkpoint="chunkformer-large-vie",
        audio_path="samples/test.wav",
        label_text=None,
        chunk_size=64,
        left_context_size=128,
        right_context_size=128,
        total_batch_duration=1800
    )

# ==================== MODEL ====================
def load_model_only(model_checkpoint="../chunkformer-large-vie", device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, char_dict = init(model_checkpoint, device)
    model.eval()
    return model, char_dict

def dump_module_structure(module: torch.nn.Module, output_path: str, title: str = ""):
    with open(output_path, "w", encoding="utf-8") as f:
        if title:
            f.write(f"==== {title} ====\n\n")
        f.write(str(module))


# ==================== PREPROCESS ====================
def prepare_input_file(audio_path, device):
    waveform = load_audio(audio_path)
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    if waveform.abs().max() <= 1.0:
        waveform = waveform * 32768.0
    waveform = waveform.clamp(-32768, 32767).to(device)

    feats = fbank(
        waveform,
        num_mel_bins=80,
        frame_length=25,
        frame_shift=10,
        dither=0.0,
        energy_floor=0.0,
        sample_frequency=16000
    ).unsqueeze(0).to(device)
    return feats

@torch.no_grad()
def decode_aed_long_form(xs, model, char_dict, args, device):
    chunk_size = args.chunk_size
    left_context_size = args.left_context_size
    right_context_size = args.right_context_size
    subsampling_factor = model.encoder.embed.subsampling_factor
    conv_kernel_size = model.encoder.cnn_module_kernel
    conv_lorder = conv_kernel_size // 2
    num_blocks = model.encoder.num_blocks
    hidden_dim = model.encoder._output_size
    attention_heads = model.encoder.attention_heads

    max_len = int(args.total_batch_duration // 0.01) // 2
    multiply_n = max_len // chunk_size // subsampling_factor
    truncated_context_size = chunk_size * multiply_n
    rel_right_context_size = (
        right_context_size + max(chunk_size, right_context_size) * (num_blocks - 1)
    ) * subsampling_factor

    # Init cache
    offset = torch.zeros(1, dtype=torch.int, device=device)
    att_cache = torch.zeros((num_blocks, left_context_size, attention_heads, hidden_dim * 2 // attention_heads), device=device)
    cnn_cache = torch.zeros((num_blocks, hidden_dim, conv_lorder), device=device)

    encoder_outs_chunks = []
    encoder_lens_total = 0

    for idx in range(0, xs.shape[1], truncated_context_size * subsampling_factor):
        start = truncated_context_size * subsampling_factor * idx
        end = min(start + truncated_context_size * subsampling_factor + 7, xs.shape[1])

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

        encoder_outs = encoder_outs[:, :encoder_lens]
        encoder_outs_chunks.append(encoder_outs)
        encoder_lens_total += encoder_outs.shape[1]

        if start + rel_right_context_size >= xs.shape[1]:
            break

    # Cat all chunk outputs
    encoder_outs_full = torch.cat(encoder_outs_chunks, dim=1)
    encoder_mask = torch.ones(1, 1, encoder_outs_full.shape[1], dtype=torch.bool, device=device)

    # Decode AED
    pred_token_ids, _ = model.decode_aed(encoder_outs_full, encoder_mask, maxlen=100)
    pred_ids = pred_token_ids[0].tolist()  # Đổi tên cho ngắn gọn

    # Raw vs Clean
    raw = transcript_raw(pred_ids, char_dict)
    clean = transcript_clean(pred_ids, char_dict)

    print(f"📣 AED RAW   : {raw}")
    print(f"📣 AED CLEAN : {clean}")

    return raw, clean

def transcript_raw(token_ids, char_dict):
    """
    Convert list of token IDs → raw string (giữ nguyên các ký tự subword, <sos>, <eos>, blank).
    Args:
        token_ids: List[int]
        char_dict: Dict[int, str]
    Returns:
        raw_transcript: str
    """
    return "".join(char_dict.get(idx, "") for idx in token_ids)


def transcript_clean(token_ids, char_dict,
                     sos_token="<sos>", eos_token="<eos>",
                     blank_token="_"):
    """
    Tạo bản clean:
      - Loại bỏ sos/eos/blank
      - Chuyển ký tự subword (‘▁’) thành khoảng trắng
    Args:
        token_ids: List[int]
        char_dict: Dict[int, str]
        sos_token: chuỗi biểu diễn <sos> trong char_dict
        eos_token: chuỗi biểu diễn <eos> trong char_dict
        blank_token: ký tự blank (thường là "_")
    Returns:
        clean_transcript: str
    """
    parts = []
    for idx in token_ids:
        c = char_dict.get(idx, "")
        if c in (sos_token, eos_token, blank_token):
            continue
        if c.startswith("▁"):
            # subword marker → space + phần còn lại
            parts.append(" " + c.lstrip("▁"))
        else:
            parts.append(c)
    return "".join(parts).strip()


@torch.no_grad()  # Không cần gradient vì đang trong chế độ inference
def decode_long_form(xs, model, char_dict, args, device):
    """
    Input:
        xs: Tensor dạng [1, T_raw, 80], đặc trưng âm thanh (FBank)
        model: mô hình ASR (ChunkFormer + CTC)
        char_dict: từ điển ánh xạ ID → ký tự
        args: các hyperparameter decode
        device: cuda hoặc cpu
    Output:
        full_text: chuỗi ký tự đã decode
    """

    def get_max_input_context(c, r, n):
        # Tính tổng số frame phải cần nhìn bên phải
        return r + max(c, r) * (n - 1)

    # === Lấy config ===
    '''
    🎯 I. Giải thích khái niệm chunk trong ASR
    🧠 1. Vấn đề:
    Mô hình ASR truyền thống cần toàn bộ đoạn audio để decode → ❌ không dùng được cho realtime.
    Càng dài, càng tốn RAM và không thể xử lý online.

    ⚡ Giải pháp: Chunk-based decoding
    Chia đặc trưng âm thanh đầu vào (xs) thành các khối nhỏ có độ dài cố định.
    Mỗi khối (chunk) được xử lý riêng biệt, nhưng vẫn có context trái/phải để mô hình hiểu ngữ cảnh.
    '''
    chunk_size = args.chunk_size                      # Số frame mỗi chunk sau subsampling
    left_context_size = args.left_context_size        # Số frame trái cho attention
    right_context_size = args.right_context_size      # Số frame phải cho attention
    subsampling_factor = model.encoder.embed.subsampling_factor  # thường = 4
    conv_lorder = model.encoder.cnn_module_kernel // 2  # lookahead của conv (thường = 7)
    '''
    | Term                     | Ý nghĩa                                                           |
    | ------------------------ | ----------------------------------------------------------------- |
    | `chunk_size`             | Số frame chính cần decode trong mỗi khối                          |
    | `left_context_size`      | Số frame bên trái được dùng làm ngữ cảnh                          |
    | `right_context_size`     | Số frame bên phải (look-ahead) hỗ trợ mô hình                     |
    | `truncated_context_size` | Tổng số frame có thể xử lý trong 1 chunk theo giới hạn tài nguyên |

    '''

    # === Lấy config từ args và model ===
    chunk_size = args.chunk_size
    left_context_size = args.left_context_size
    right_context_size = args.right_context_size
    subsampling_factor = model.encoder.embed.subsampling_factor
    conv_kernel_size = model.encoder.cnn_module_kernel
    conv_lorder = conv_kernel_size // 2
    num_blocks = model.encoder.num_blocks
    hidden_dim = model.encoder._output_size
    attention_heads = model.encoder.attention_heads

    # print("========== 🔧 CONFIG SUMMARY ==========")
    # print(f"  chunk_size               = {chunk_size}")
    # print(f"  left_context_size        = {left_context_size}")
    # print(f"  right_context_size       = {right_context_size}")
    # print(f"  subsampling_factor       = {subsampling_factor}")
    # print(f"  conv_kernel_size         = {conv_kernel_size}")
    # print(f"  conv_lorder              = {conv_lorder}")
    # print(f"  num_blocks               = {num_blocks}")
    # print(f"  hidden_dim (_output_size)= {hidden_dim}")
    # print(f"  attention_heads          = {attention_heads}")
    # print(f"  Input shape              = {xs.shape}")  # [1, T_raw, 80]
    # print("=======================================\n")

    # === Tính độ dài thực tế mỗi chunk có thể xử lý trong GPU ===
    max_len = int(args.total_batch_duration // 0.01) // 2  # Tổng số frame (trước subsample)
    multiply_n = max_len // chunk_size // subsampling_factor
    truncated_context_size = chunk_size * multiply_n  # số frame giữ lại mỗi chunk sau encoder

    # === Tính tổng context phải (phục vụ attention + conv) sau subsample ===
    rel_right_context_size = get_max_input_context(
        chunk_size, max(right_context_size, conv_lorder), model.encoder.num_blocks
    ) * subsampling_factor


    # print("🧮 CALCULATED CONTEXT INFO")
    # print(f"  total_batch_duration     = {args.total_batch_duration}")
    # print(f"  max_len (frame count)    = {max_len}")
    # print(f"  multiply_n               = {multiply_n}")
    # print(f"  truncated_context_size   = {truncated_context_size}")
    # print(f"  rel_right_context_size   = {rel_right_context_size}\n")
    

    '''
    🧱 Bước 1: Tách theo chunk
    text
    Copy code
    🔣 Full Input xs: [1, 1000, 80]
        │
        ├──> Chunk 0: [start=0 : end=256] + rel_right_context=64 → [1, 320, 80]
        │
        ├──> Chunk 1: [start=256 : end=512] + rel_right_context=64 → [1, 320, 80]
        │
        └──> ...
    Mỗi chunk lấy chính xác chunk_size * subsampling_factor * N

    Phía sau cộng thêm rel_right_context_size để mô hình có look-ahead  
    '''

    # === Init cache: cho attention & conv, lưu trạng thái giữa các chunk ===
    offset = torch.zeros(1, dtype=torch.int, device=device)
    num_blocks = model.encoder.num_blocks
    att_cache = torch.zeros(
                    (
                        num_blocks, 
                        left_context_size,
                        model.encoder.attention_heads,
                        model.encoder._output_size * 2 // model.encoder.attention_heads
                    ), 
                    device=device
                )
    cnn_cache = torch.zeros(
                    (
                        num_blocks, 
                        model.encoder._output_size, 
                        conv_lorder
                    ), 
                    device=device
                )

    # === Danh sách chứa kết quả dự đoán theo frame (ID nhãn) của từng chunk ===
    framewise_token_ids_chunks = []

    # === Duyệt qua toàn bộ input xs (T_raw frame) theo từng chunk ===
    num_chunks = 0
    for idx in range(0, xs.shape[1], truncated_context_size * subsampling_factor):
        start = truncated_context_size * subsampling_factor * idx
        end = min(start + truncated_context_size * subsampling_factor + 7, xs.shape[1])

        # x: [1, T_chunk + context_right, 80]
        x = xs[:, start:end + rel_right_context_size]
        x_len = torch.tensor([x[0].shape[0]], dtype=torch.int, device=device)  # [1]

        # === Forward encoder theo chunk (streaming mode) ===
        '''
        📤 Bước 3: Chạy CTC trên chunk_output
        text
        Copy code
        encoder_out: [1, 64, 512]
            │
            └──> CTC Linear → logits: [1, 64, vocab]
                        ↓
                    Argmax → [64]  ← framewise_token_ids (ID nhãn)
        Mỗi chunk → chuỗi ID token dự đoán cho 64 frame.
        '''
        # print(f"\n📦 Chunk {num_chunks}")
        # print(f"  Input chunk frame idx: {start} → {end}")
        # print(f"  Input x shape         : {x.shape}")
        # print(f"  x_len                 : {x_len.tolist()}")


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
        # encoder_outs: [1, T_out, 512], encoder_lens: [1]
        # print(f"  encoder_outs shape    : {encoder_outs.shape}")
        # print(f"  encoder_lens          : {encoder_lens.tolist()}")

        # === Reshape và cắt bỏ phần rel_right_context thừa ===
        encoder_outs = encoder_outs.reshape(1, -1, encoder_outs.shape[-1])[:, :encoder_lens]
        if chunk_size * multiply_n * subsampling_factor * idx + rel_right_context_size < xs.shape[1]:
            encoder_outs = encoder_outs[:, :truncated_context_size]  # giữ lại phần chính

        offset = offset - encoder_lens + encoder_outs.shape[1]

        # === Chạy CTC forward: từ [1, T_out, 512] → [T_out] token_id ===
        framewise_ids = model.encoder.ctc_forward(encoder_outs).squeeze(0)  # shape: [T_out]
        # print(f"  framewise_ids shape   : {framewise_ids.shape}")
        # print(f"  framewise_ids (first 10): {framewise_ids.tolist()[:10]}")
        
        framewise_token_ids_chunks.append(framewise_ids)
        num_chunks += 1

        # === Nếu đã xử lý xong toàn bộ input thì dừng ===
        if start + rel_right_context_size >= xs.shape[1]:
            break

    # === Gộp các chunk lại: [T_total]
    '''
    📦 Bước 4: Gom lại toàn bộ các chunk
    text
    Copy code
    framewise_token_ids_chunks = [
        [id_1, id_2, ..., id_64],       ← Chunk 0
        [id_65, ..., id_128],           ← Chunk 1
        ...
    ]
    → cat → full_framewise_ids: [total_T]
    '''
    full_framewise_ids = torch.cat(framewise_token_ids_chunks)
    print(f"\n📊 Total full_framewise_ids shape: {full_framewise_ids.shape}")
    print(f"    Sample token IDs (first 20): {full_framewise_ids.tolist()[:20]}")

    # === Decode theo CTC: loại blank, duplicate, convert ID → char, gán timestamp ===
    '''
    🔄 Bước 5: Decode CTC
    text
    Copy code
    full_framewise_ids = [b, b, a, a, _, b, b, _]
            ↓
        Remove dup: [b, a, _, b]
        Remove blank: [b, a, b]
        Map ID → char
        Add timestamps per chunk
    '''
    decoded_segments = get_output_with_timestamps([full_framewise_ids], char_dict)[0]
    print("\n📝 Decoded segments (first 3):")
    for seg in decoded_segments[:3]:
        print(f"  → {seg}")

    # === Ghép tất cả lại thành câu hoàn chỉnh ===
    final_transcript = " ".join([item["decode"] for item in decoded_segments])
    print(f"\n✅ Final transcript: {final_transcript}")
    return final_transcript


# ==================== MAIN ====================
def run_inference(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, char_dict = init(args.model_checkpoint, device)
    model.eval()

    feats = prepare_input_file(args.audio_path, device)
    decoded = decode_long_form(feats, model, char_dict, args, device)

    print("🟢 Prediction     :", decoded)
    if args.label_text:
        print("🔵 Ground Truth   :", args.label_text)
        print("❌ WER            :", wer(args.label_text.lower(), decoded.lower()))

if __name__ == '__main__':
    args = get_args()
    run_inference(args)

'''
python -m chunkformer_vpb.model_utils \
  --model_checkpoint ../chunkformer-large-vie \
  --audio_path ../debug_wavs/sample_00.wav \
  --label_text "nửa vòng trái đất hơn bảy năm"

'''
