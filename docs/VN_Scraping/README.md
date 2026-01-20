# VN Text Scraper

Minimal scraper to extract and clean text from Visual Novels for LLM training.

## Setup

1. **Install Python** (3.7+)

2. **Create folder structure**:
   ```
   raw_data/     # Copy game files here
   cleaned/      # Output goes here (staging)
   ```
   After cleaning, move the final training file to:
   `training_data/vn/cleaned_binary_dialogue.txt` (gitignored).

3. **Copy game files**:
   - Copy the "Dies irae DX package" files into `raw_data/`
   - Focus on DAT files and script files

## Usage

```bash
python src/scraper.py
```

## Learning Path

### Step 1: Inspect Files
- Open DAT files in a hex editor (HxD, 010 Editor)
- Look for text strings
- Identify file format/encoding

### Step 2: Extract Archives
- Use GARbro or similar tool to extract DAT files
- Find script/text files inside

### Step 3: Identify Text Format
- Open script files in text editor
- Find dialogue patterns
- Note character name format

### Step 4: Modify Scraper
- Update `extract_dialogue()` with your patterns
- Adjust `clean_text()` for your format
- Test on one file first

### Step 5: Process All Files
- Run scraper on all files
- Review cleaned output
- Adjust cleaning rules as needed

## Next Steps

After cleaning:
1. Tokenize with GPT-2 tiktokenizer
2. Feed into your pre-trained LLM
3. Fine-tune model

## Resources

- **GARbro**: https://github.com/morkt/GARbro
- **Hex Editor**: HxD (free)
- **Encoding**: Check file headers for encoding hints

