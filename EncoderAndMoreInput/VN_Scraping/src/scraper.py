"""
Minimal VN Text Scraper
Extracts and cleans text from Visual Novel files for LLM training.
"""

import os
import re
from pathlib import Path


def read_file(file_path):
    """Read a file and return its contents.
    
    Try different encodings common in VNs:
    - UTF-8
    - Shift-JIS (Japanese)
    - UTF-16
    """
    encodings = ['utf-8', 'shift_jis', 'utf-16', 'latin-1']
    
    for encoding in encodings:
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                return f.read()
        except (UnicodeDecodeError, UnicodeError):
            continue
    
    # If text fails, try binary
    with open(file_path, 'rb') as f:
        return f.read()


def extract_dialogue(text):
    """Extract dialogue from text.
    
    This is a placeholder - you'll need to identify
    the actual dialogue patterns in your VN files.
    
    Common patterns:
    - Character name followed by colon
    - Text in quotes (Japanese or English)
    - Lines after specific markers
    """
    dialogue_lines = []
    
    # Split into lines
    lines = text.split('\n')
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # Skip metadata/commands
        if line.startswith('DZI') or line.startswith('bg_') or ',' in line and '\\' in line:
            continue
        
        # Try various dialogue patterns
        # Japanese quotes: 「text」 or 『text』
        jp_quotes = re.findall(r'[「『]([^」』]+)[」』]', line)
        if jp_quotes:
            dialogue_lines.extend(jp_quotes)
            continue
        
        # English quotes
        en_quotes = re.findall(r'["\']([^"\']+)["\']', line)
        if en_quotes:
            dialogue_lines.extend(en_quotes)
            continue
        
        # Character: Dialogue format
        if ':' in line and len(line.split(':')) == 2:
            char, dialogue = line.split(':', 1)
            if dialogue.strip():
                dialogue_lines.append(dialogue.strip())
            continue
        
        # If line has Japanese characters and is substantial, include it
        if re.search(r'[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FAF]', line) and len(line) > 5:
            dialogue_lines.append(line)
            continue
        
        # If line looks like dialogue (has quotes or is substantial text)
        if len(line) > 10 and not line.startswith('[') and not line.startswith('@'):
            dialogue_lines.append(line)
    
    return dialogue_lines


def clean_text(text):
    """Clean extracted text for LLM training.
    
    Removes:
    - Game commands [like this]
    - Control codes
    - Extra whitespace
    - Special formatting
    """
    # Remove game commands in brackets
    text = re.sub(r'\[.*?\]', '', text)
    
    # Remove control codes (common in VNs)
    text = re.sub(r'\\[a-z]+\d*', '', text)
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    
    return text


def process_file(input_path, output_path):
    """Process a single file: extract and clean text."""
    print(f"Processing: {input_path}")
    
    # Skip known non-text files
    if input_path.suffix.lower() in ['.png', '.jpg', '.bmp', '.gif', '.dll', '.exe']:
        print(f"  Skipping non-text file")
        return
    
    # Skip DZI/metadata files
    try:
        with open(input_path, 'rb') as f:
            header = f.read(10)
            if header.startswith(b'DZI'):
                print(f"  Skipping DZI metadata file")
                return
    except:
        pass
    
    # Read file
    content = read_file(input_path)
    
    # If binary, you'll need to parse it differently
    if isinstance(content, bytes):
        print(f"  Binary file - needs special parsing")
        return
    
    # Extract dialogue (adjust based on format)
    dialogue = extract_dialogue(content)
    
    # Clean each line
    cleaned_lines = [clean_text(line) for line in dialogue if line.strip()]
    
    # Write output
    with open(output_path, 'w', encoding='utf-8') as f:
        for line in cleaned_lines:
            if line:  # Skip empty lines
                f.write(line + '\n')
    
    print(f"  Extracted {len(cleaned_lines)} lines")


def main():
    """Main function - entry point."""
    import sys
    
    # If a text file is provided, process it directly
    if len(sys.argv) > 1 and Path(sys.argv[1]).is_file():
        # Process a single text file (like final_dialogue.txt)
        input_file = Path(sys.argv[1])
        output_file = Path('cleaned') / input_file.stem / 'cleaned.txt'
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        print(f"Processing text file: {input_file}")
        process_file(input_file, output_file)
        return
    
    # Setup paths - allow command line argument
    if len(sys.argv) > 1:
        raw_dir = Path(sys.argv[1])
    else:
        raw_dir = Path('raw_data')
    
    output_dir = Path('cleaned')
    output_dir.mkdir(exist_ok=True)
    
    # Process files
    if not raw_dir.exists():
        print(f"Directory not found: {raw_dir}")
        print(f"\nUsage:")
        print(f"  python src/scraper.py [path_to_data_folder]")
        print(f"  python src/scraper.py [path_to_text_file]  # Process single file")
        print(f"\nExample:")
        print(f"  python src/scraper.py \"C:\\Users\\bense\\Downloads\\MG1041_1\\Dies irae DX package\\data\"")
        print(f"  python src/scraper.py final_dialogue.txt")
        return
    
    # Find all files to process
    files = list(raw_dir.rglob('*'))
    
    for file_path in files:
        if file_path.is_file():
            # Create output path
            rel_path = file_path.relative_to(raw_dir)
            output_path = output_dir / rel_path.with_suffix('.txt')
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Process file
            try:
                process_file(file_path, output_path)
            except Exception as e:
                print(f"Error processing {file_path}: {e}")


if __name__ == '__main__':
    main()

