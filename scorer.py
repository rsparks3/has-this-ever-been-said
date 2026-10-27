"""Language-model scoring for "has this ever been said?".

A small causal LM assigns every token in the text a probability given the tokens before it.
The product of those probabilities is P(text): roughly, the chance that a random stretch of
human language beginning at a sentence boundary is exactly this text. The dashboard multiplies
that by how much language humanity has produced to estimate how many times the text has
already been said or written.

No end-of-sequence token is scored, so P(text) is the probability of the text appearing as a
prefix/substring, not of someone stopping exactly there. That is what we want.
"""
from __future__ import annotations

import math
import os
import threading
import time
from dataclasses import dataclass, field

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = os.environ.get("EVERSAID_MODEL", "HuggingFaceTB/SmolLM2-360M")
MAX_TOKENS = int(os.environ.get("EVERSAID_MAX_TOKENS", "512"))
LN10 = math.log(10.0)


@dataclass
class TokenScore:
    text: str            # the token as text, leading space preserved
    log10p: float        # log10 P(token | previous tokens)
    rank: int            # 1 = the model's top guess
    expected: str        # what the model's top guess was
    expected_log10p: float
    word_index: int      # which whitespace-delimited word this token belongs to (0-based)

    def as_dict(self) -> dict:
        return {
            "text": self.text,
            "log10p": self.log10p,
            "bits": -self.log10p * LN10 / math.log(2.0),
            "rank": self.rank,
            "expected": self.expected,
            "expected_log10p": self.expected_log10p,
            "word": self.word_index,
        }


@dataclass
class TextScore:
    tokens: list[TokenScore]
    log10p: float                 # log10 P(whole text)
    words: list[str] = field(default_factory=list)
    word_log10p: list[float] = field(default_factory=list)   # cumulative log10 P at the end of each word
    seconds: float = 0.0

    def as_dict(self) -> dict:
        n = len(self.tokens)
        return {
            "model": MODEL_ID,
            "n_tokens": n,
            "n_words": len(self.words),
            "log10p": self.log10p,
            "bits_per_token": (-self.log10p * LN10 / math.log(2.0) / n) if n else 0.0,
            "tokens": [t.as_dict() for t in self.tokens],
            "words": self.words,
            "word_log10p": self.word_log10p,
            "seconds": self.seconds,
        }


class Scorer:
    """Loads the model lazily on a background thread; `ready` flips when scoring can start."""

    def __init__(self, model_id: str = MODEL_ID, device: str = "cpu") -> None:
        self.model_id = model_id
        self.device = device
        self.tokenizer = None
        self.model = None
        self.ready = False
        self.error: str | None = None
        self._lock = threading.Lock()
        self._thread = threading.Thread(target=self._load, daemon=True)
        self._thread.start()

    def _load(self) -> None:
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            self.model = AutoModelForCausalLM.from_pretrained(self.model_id, dtype=torch.float32)
            self.model.to(self.device).eval()
            self.ready = True
        except Exception as exc:  # surfaced through /api/info so the page can show it
            self.error = f"{type(exc).__name__}: {exc}"

    def _bos_id(self) -> int | None:
        tok = self.tokenizer
        return tok.bos_token_id if tok.bos_token_id is not None else tok.eos_token_id

    @torch.no_grad()
    def score(self, text: str) -> TextScore:
        if not self.ready:
            raise RuntimeError("model not loaded yet")
        tok, model = self.tokenizer, self.model
        ids = tok(text, add_special_tokens=False)["input_ids"][:MAX_TOKENS]
        if not ids:
            return TextScore(tokens=[], log10p=0.0)

        t0 = time.time()
        bos = self._bos_id()
        if bos is None:   # extremely unusual; fall back to a newline as "start of text" context
            bos = tok("\n", add_special_tokens=False)["input_ids"][0]
        input_ids = torch.tensor([[bos] + ids], device=self.device)
        with self._lock:
            logits = model(input_ids=input_ids).logits[0].float()
        # Row i of logits predicts input position i+1, so rows 0..n-1 predict the n text tokens.
        pred = logits[: len(ids)]
        targets = torch.tensor(ids, device=self.device)
        ids_scored = ids
        logp = torch.log_softmax(pred, dim=-1)
        tok_logp = logp.gather(1, targets[:, None])[:, 0]
        ranks = (logp > tok_logp[:, None]).sum(dim=1) + 1
        top_ids = logp.argmax(dim=1)
        top_logp = logp.gather(1, top_ids[:, None])[:, 0]

        pieces = [tok.decode([i], clean_up_tokenization_spaces=False) for i in ids_scored]
        top_pieces = [tok.decode([int(i)], clean_up_tokenization_spaces=False) for i in top_ids]

        tokens: list[TokenScore] = []
        words: list[str] = []
        word_log10p: list[float] = []
        cum = 0.0
        for k, piece in enumerate(pieces):
            starts_word = k == 0 or piece[:1].isspace() or (pieces[k - 1].endswith("\n"))
            if starts_word:
                words.append(piece.strip())
                word_log10p.append(0.0)
            else:
                words[-1] += piece
            l10 = float(tok_logp[k]) / LN10
            cum += l10
            word_log10p[-1] = cum
            tokens.append(TokenScore(
                text=piece, log10p=l10, rank=int(ranks[k]), expected=top_pieces[k],
                expected_log10p=float(top_logp[k]) / LN10, word_index=len(words) - 1,
            ))
        return TextScore(tokens=tokens, log10p=cum, words=words, word_log10p=word_log10p,
                         seconds=time.time() - t0)
