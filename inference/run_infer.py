import torch
from torchaudio.compliance.kaldi import fbank
from torch.utils.benchmark import Timer

from ..decode import init, load_audio
from ..model.utils.ctc_utils import get_output

# === Config ===
class Args:
    model_checkpoint = "chunkformer-large-vie"
    total_batch_duration = 1800
    chunk_size = 64
    left_context_size = 128
    right_context_size = 128
    long_form_audio = "stt_chunkformer/data/common_voice_vi_23397238.wav"

args = Args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Load model and audio ===
model, char_dict = init(args.model_checkpoint, device)
model = model.to(device)
model.eval()

waveform = load_audio(args.long_form_audio)
xs = fbank(
    waveform,
    num_mel_bins=80,
    frame_length=25,
    frame_shift=10,
    dither=0.0,
    energy_floor=0.0,
    sample_frequency=16000
).unsqueeze(0).to(device)

# === Define benchmark function ===
def run():
    offset = torch.zeros(1, dtype=torch.int, device=device)
    chunk_size = args.chunk_size
    left_context_size = args.left_context_size
    right_context_size = args.right_context_size

    x_len = torch.tensor([xs[0].shape[0]], dtype=torch.int).to(device)
    att_cache = torch.zeros((model.encoder.num_blocks, left_context_size, model.encoder.attention_heads, model.encoder._output_size * 2 // model.encoder.attention_heads)).to(device)
    cnn_cache = torch.zeros((model.encoder.num_blocks, model.encoder._output_size, model.encoder.cnn_module_kernel // 2)).to(device)

    encoder_outs, encoder_lens, _, att_cache, cnn_cache, offset = model.encoder.forward_parallel_chunk(
        xs=xs,
        xs_origin_lens=x_len,
        chunk_size=chunk_size,
        left_context_size=left_context_size,
        right_context_size=right_context_size,
        att_cache=att_cache,
        cnn_cache=cnn_cache,
        truncated_context_size=chunk_size * 10,  # 10 chunks
        offset=offset
    )

    hyp = model.encoder.ctc_forward(encoder_outs).squeeze(0).argmax(dim=-1)
    _ = get_output([hyp], char_dict)

# === Run benchmark ===
t = Timer(
    stmt="run()",
    setup="from __main__ import run",
    globals=globals()
)

print(t.timeit(5))  # Run 5 iterations and print timing
