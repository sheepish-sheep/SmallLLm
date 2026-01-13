# VN Text Extraction Plan

## Overview
Extract and clean text from Dies irae DX VN for LLM training. The cleaned text will be tokenized using GPT-2 tiktokenizer and fed into a pre-trained LLM.

## File Structure
```
VN_Scraping/
├── src/              # Scraper code
├── raw_data/         # Original game files (copy here)
├── extracted/        # Extracted text (before cleaning)
├── cleaned/          # Cleaned text (ready for tokenization)
└── output/           # Final processed files
```

Note: move the final cleaned training file to `training_data/vn/cleaned_binary_dialogue.txt` (gitignored).

## Step-by-Step Plan

### Phase 1: File Analysis
1. **Identify file types**: DAT files are likely archives
2. **Test extraction tools**: Try common VN extractors:
   - `vgmstream` for audio
   - `GARbro` or `AssetStudio` for archives
   - `HxD` or hex editor to inspect file headers
3. **Find text encoding**: VNs often use Shift-JIS, UTF-8, or UTF-16

### Phase 2: Text Extraction
1. **Extract DAT archives**:
   - Use appropriate tool for the archive format
   - Look for script files (.txt, .ks, .scn, etc.)
2. **Parse script files**:
   - Identify dialogue markers
   - Extract character names and dialogue
3. **Handle special formats**:
   - Remove game commands (e.g., `[wait]`, `[bgm]`)
   - Extract only readable text

### Phase 3: Text Cleaning
1. **Remove game commands**: Strip `[tags]`, control codes
2. **Normalize whitespace**: Clean up extra spaces/newlines
3. **Remove metadata**: Skip file headers, version info
4. **Character name separation**: Format as "Character: Dialogue"
5. **Remove duplicates**: Skip repeated lines

### Phase 4: Output Format
1. **Plain text files**: One file per scene/chapter
2. **UTF-8 encoding**: Standard for tokenization
3. **Line breaks**: One dialogue per line (or paragraph)

## Tools to Research
- **GARbro**: Universal VN archive extractor (see `GARBRO_GUIDE.md` for usage)
- **KrkrExtract**: For KiriKiri engine games
- **Python libraries**: `struct`, `zlib`, `lzma` for binary parsing
- **Hex editors**: To inspect file formats

## Minimal Code Approach
Start with:
1. File reader (binary/text)
2. Simple pattern matching for dialogue
3. Basic text cleaning functions
4. Output writer

Build incrementally - test each step before moving forward.



