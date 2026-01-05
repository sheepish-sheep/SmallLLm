import tiktoken

HL_START = "<hl>"
HL_END = "</hl>"
HL_START_ID = 50257
HL_END_ID = 50258


def build_hl_encoding():
    """
    Build a GPT-2-compatible encoding with highlight special tokens.
    """
    base = tiktoken.get_encoding("gpt2")
    special = dict(base._special_tokens)
    special[HL_START] = HL_START_ID
    special[HL_END] = HL_END_ID
    return tiktoken.Encoding(
        name="gpt2_hl",
        pat_str=base._pat_str,
        mergeable_ranks=base._mergeable_ranks,
        special_tokens=special,
    )


def encode_with_hl(enc, text: str):
    """
    Encode text while keeping HL tokens intact.
    """
    return enc.encode(text, allowed_special={HL_START, HL_END})


def decode_with_hl(enc, ids):
    """
    Decode ids back to text and keep HL tokens intact.
    """
    return enc.decode(ids)

