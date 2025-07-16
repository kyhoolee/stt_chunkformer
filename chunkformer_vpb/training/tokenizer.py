
from torch.nn.utils.rnn import pad_sequence
from typing import Dict, List, Tuple
from typing import Tuple
import torch

# modules/textnorm.py
import re
import unicodedata

_VI_ACCENT_RE = re.compile(r"[̣̀́̉̃]")
_PUNC_TABLE   = str.maketrans({
    "–": "-", "—": "-", "“": "\"", "”": "\"", "‘": "'", "’": "'",
})

def normalize_vi(text: str) -> str:
    # 1) Unicode chuẩn
    text = unicodedata.normalize("NFC", text)
    # 2) Chuẩn hoá punctuation
    text = text.translate(_PUNC_TABLE)
    # 3) Lowercase + strip spaces
    text = " ".join(text.lower().split())
    return text

# ==================== TOKENIZER UTILS ====================

class GreedyTokenizer:
    def __init__(self, vocab_path: str):
        """
        vocab_path: đường dẫn tới file vocab.txt (mỗi dòng 1 token)
        """
        # ---------- load vocab ----------
        self.vocab = []
        with open(vocab_path, encoding="utf-8") as f:
            for line in f:
                token = line.strip().split()[0]
                self.vocab.append(token)

        # ---------- basic map ----------
        self.token2id: Dict[str, int] = {tok: idx for idx, tok in enumerate(self.vocab)}
        self.id2token: List[str] = self.vocab
        # sort decreasing độ dài để greedy match khi encode
        self.vocab_sorted = sorted(self.vocab, key=len, reverse=True)

        # ---------- special IDs ----------
        # giả định chuẩn Wenet: 0 = <blank> = pad, cuối vocab = <sos/eos>
        self.blank_id = 0
        self.pad_id   = 0                    # pad dùng cùng giá trị với blank
        self.sos_id   = len(self.vocab) - 1  # eos = sos
        self.eos_id   = self.sos_id

    # ---------- ENCODE ----------
    def tokenize(self, text: str) -> List[int]:
        """
        Greedy longest-match tokenization.
        - Thêm '▁' đầu câu, thay space bằng '▁'.
        """
        s = "▁" + text.strip().replace(" ", "▁")
        idx, L, ids = 0, len(s), []
        while idx < L:
            for tok in self.vocab_sorted:
                if s.startswith(tok, idx):
                    ids.append(self.token2id[tok])
                    idx += len(tok)
                    break
            else:
                # fallback: <unk> nếu có, hoặc skip 1 char
                ids.append(self.token2id.get("<unk>", self.blank_id))
                idx += 1
        return ids

    # ---------- DECODE ----------
    def decode_ids(self, ids: List[int]) -> str:
        """
        Convert list[int] → raw string, bỏ blank/pad/sos/eos.
        """
        tokens = [
            self.id2token[i]
            for i in ids
            if i not in {self.blank_id, self.pad_id, self.sos_id, self.eos_id}
        ]
        text = "".join(tokens).replace("▁", " ").strip()
        return text

    # ---------- helper cho AED ----------
    def text2labels(self, text: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return ys_pad & ys_lens (đã thêm sos/eos) để dùng cho AED.
        """
        ids = self.tokenize(text)
        seq = [self.sos_id] + ids + [self.eos_id]
        ys_pad  = torch.tensor(seq, dtype=torch.long).unsqueeze(0)   # [1, L]
        ys_lens = torch.tensor([len(seq)], dtype=torch.long)
        return ys_pad, ys_lens


    def build_y_batch(
        self,
        toks: torch.Tensor,       # [B, L_pad] no sos/eos
        tok_lens: torch.Tensor,   # [B]
        sos_id: int,
        eos_id: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Thêm sos/eos cho mỗi dòng, rồi pad về batch.
        Trả về:
          ys_pad:  [B, L_max+2]
          ys_lens: [B]
        """
        seqs = []
        lens = []
        B = toks.size(0)
        for i in range(B):
            raw = toks[i, : tok_lens[i]].tolist()
            seq = [sos_id] + raw + [eos_id]
            seqs.append(torch.tensor(seq, dtype=torch.long))
            lens.append(len(seq))
        ys_pad = pad_sequence(seqs, batch_first=True, padding_value=self.token2id.get("<pad>", 0))
        ys_lens = torch.tensor(lens, dtype=torch.long)
        return ys_pad, ys_lens


