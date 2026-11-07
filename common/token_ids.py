"""Checkpoint-level token conventions, shared by pruning and evaluation."""
import json
from pathlib import Path


DEFAULT_MASK_TOKEN_ID = 151669


def validate_tokenizer_vocab(tokenizer, draft_tokenizer, *, vocab_size: int, mask_token_id: int) -> None:
    """Require one token-ID space for data, draft embeddings and verification."""
    vocab = tokenizer.get_vocab()
    draft_vocab = draft_tokenizer.get_vocab()
    # Published D3D checkpoints add a dedicated mask token to Qwen3's vocabulary.
    for tok, mapping in ((tokenizer, vocab), (draft_tokenizer, draft_vocab)):
        if getattr(tok, "mask_token_id", None) == int(mask_token_id):
            mapping.pop(tok.mask_token, None)
    if vocab != draft_vocab:
        raise ValueError(
            "Target and draft tokenizer IDs differ. Prepare the draft with "
            "python -m training.prepare_draft before training or inference."
        )
    ids = set(vocab.values())
    if not ids or min(ids) < 0 or max(ids) >= int(vocab_size):
        raise ValueError("Tokenizer IDs are outside the draft model vocabulary")
    reserved_mask = any(
        value == int(mask_token_id) and name.startswith("<|reserved_special_token_") and name.endswith("|>")
        for name, value in vocab.items()
    )
    if not 0 <= int(mask_token_id) < int(vocab_size) or (int(mask_token_id) in ids and not reserved_mask):
        raise ValueError("mask_token_id must be an unused row or a reserved special token")


def checkpoint_mask_id(checkpoint, override: int | None = None) -> int:
    config = json.loads((Path(checkpoint) / "config.json").read_text())
    saved = config.get("mask_token_id")
    if saved is not None and override is not None and int(saved) != int(override):
        raise ValueError("--mask-token-id conflicts with the checkpoint's mask_token_id")
    value = override if override is not None else saved
    value = DEFAULT_MASK_TOKEN_ID if value is None else int(value)
    if not 0 <= value < int(config["vocab_size"]):
        raise ValueError(f"mask token ID {value} is outside checkpoint vocabulary {config['vocab_size']}")
    return value
