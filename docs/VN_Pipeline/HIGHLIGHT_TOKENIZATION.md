# Highlight Tokenization

This document explains how `<hl>` and `</hl>` tokens work in the pipeline.

## Why Special Tokens?

Regular text like `<hl>` would be tokenized as multiple tokens:
- `<` → token 27
- `hl` → token 5765
- `>` → token 29

This breaks the model's ability to understand that `<hl>` is a single semantic unit.

**Solution**: Register `<hl>` and `</hl>` as **special tokens** with dedicated IDs.

## Implementation

File: `EncoderAndMoreInput/VN_Pipeline/utils/specialtoken_hl.py`

```python
HL_START = "<hl>"
HL_END = "</hl>"
HL_START_ID = 50257  # First ID after GPT-2's 50257 tokens
HL_END_ID = 50258

def build_hl_encoding():
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
```

## Usage

### Encoding with Highlights

```python
from EncoderAndMoreInput.VN_Pipeline.utils.specialtoken_hl import build_hl_encoding

enc = build_hl_encoding()

# Encode text with highlights
text = "The <hl>quick</hl> brown fox"
tokens = enc.encode(text, allowed_special={"<hl>", "</hl>"})
# Result: [464, 50257, 29782, 50258, 7586, 21831]
#          The  <hl>   quick  </hl>  brown  fox
```

### Decoding

```python
decoded = enc.decode(tokens)
# Result: "The <hl>quick</hl> brown fox"
```

## Token IDs

| Token | ID | Notes |
|-------|-----|-------|
| GPT-2 vocab | 0-50256 | Standard GPT-2 tokens |
| `<\|endoftext\|>` | 50256 | GPT-2's special token |
| `<hl>` | 50257 | Highlight start |
| `</hl>` | 50258 | Highlight end |

## Vocab Size

With highlight tokens, use `vocab_size = 50304` (or at least 50259).

The model's embedding layer must be large enough:

```python
# In config.json
{
    "vocab_size": 50304
}
```

## Where Highlights Are Used

1. **data_prep_stub.py**: Creates training pairs with highlighted spans
2. **seq2seq_train_stub.py**: Tokenizes source/target with highlights
3. **seq2seq_infer.py**: Encodes input and strips highlights from output

## Example Training Pair

```json
{
    "source": "SOURCE: Narrator: The <hl>door</hl> creaked open. => TARGET: ",
    "target": "Narrator: The door creaked open."
}
```

The model learns:
- Everything outside `<hl>...</hl>` should be copied exactly
- The highlighted span may be modified (or copied as-is for copy training)

