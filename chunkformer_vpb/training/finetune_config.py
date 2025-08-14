# modules/finetune_config.py

import yaml
from dataclasses import dataclass, field
from typing import Dict, Any
from .tokenizer import GreedyTokenizer

@dataclass
class DataConfigFT:
    manifest_dir: str
    audio_dir: str = None  # nếu không có thì lấy từ manifest_dir
    sample_rate: int = 16000
    num_mel_bins: int = 80
    frame_length: float = 25.0    # ms
    frame_shift: float = 10.0     # ms
    dither: float = 0.0
    energy_floor: float = 0.0

    train_meta_file: str = "train_meta.json"  # tên file manifest cho tập train
    valid_meta_file: str = "valid_meta.json"  # tên file manifest cho tập valid

@dataclass
class TokenizerConfigFT:
    vocab_path: str
    # tokenzier instance sẽ được gắn thêm
    tokenizer: GreedyTokenizer = field(init=False)

    def init_tokenizer(self):
        self.tokenizer = GreedyTokenizer(self.vocab_path)
        vocab = []
        with open(self.vocab_path, encoding="utf-8") as f:
            for line in f:
                token = line.strip().split()[0]
                vocab.append(token)

        # 2. Build mapping and sorted vocab
        self.token2id = {token: idx for idx, token in enumerate(vocab)}
        self.vocab = vocab
        # self.vocab_sorted = sorted(vocab, key=len, reverse=True)

@dataclass
class ModelConfigFT:
    checkpoint: str
    ctc_weight: float = 0.5

@dataclass
class TrainingConfigFT:
    batch_size: int       = 4
    lr: float             = 1e-4
    weight_decay: float   = 0.01
    warmup_steps: int     = 1000
    epochs: int           = 10
    max_grad_norm: float  = 5.0
    log_steps: int        = 50
    checkpoint_dir: str   = "checkpoints"
    shuffle: bool = True          #  ← thêm dòng này

@dataclass
class ChunkConfigFT:
    chunk_size: int
    left_context_size: int
    right_context_size: int
    total_batch_duration: int  # ms

@dataclass
class FreezeConfigFT:
    cmvn: bool = False
    subsampling: bool = False
    post_embed_norm: bool = False
    encoder_layers: int = 0
    ctc: bool = False


@dataclass
class FinetuneConfig:
    data: DataConfigFT
    tokenizer: TokenizerConfigFT
    model: ModelConfigFT
    training: TrainingConfigFT
    chunk: ChunkConfigFT
    freeze: FreezeConfigFT = None  # 👈 Thêm dòng này

    @staticmethod
    def from_yaml(path: str) -> "FinetuneConfig":
        raw = yaml.safe_load(open(path, "r"))

        data_cfg = DataConfigFT(**raw["data"])
        tok_cfg  = TokenizerConfigFT(**raw["tokenizer"])
        model_cfg = ModelConfigFT(**raw["model"])
        train_cfg = TrainingConfigFT(**raw["training"])
        chunk_cfg = ChunkConfigFT(**raw["chunk"])
        

        # init tokenizer instance
        tok_cfg.init_tokenizer()

        freeze_cfg = None
        if "freeze" in raw:
            freeze_cfg = FreezeConfigFT(**raw["freeze"])

        return FinetuneConfig(
            data=data_cfg,
            tokenizer=tok_cfg,
            model=model_cfg,
            training=train_cfg,
            chunk=chunk_cfg,
            freeze=freeze_cfg
        )

