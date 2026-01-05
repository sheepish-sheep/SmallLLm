"""
Simple Working Example - File Scraper
This is a complete, working example you can use as a starting point.
"""

import re
from pathlib import Path
import sys


def read_file(file_path: str) -> str:
    """
    Read a file with encoding fallback.
    
    Tries multiple encodings to handle different file types.
    """
    encodings = ['utf-8', 'shift_jis', 'latin-1', 'cp1252']
    
    for encoding in encodings:
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                return f.read()
        except (UnicodeDecodeError, UnicodeError):
            continue
    
    # If all text encodings fail, raise error
    raise ValueError(f"Could not decode {file_path} with any text encoding")


def extract_dialogue(text: str) -> list:
    """
    Extract dialogue from text using various patterns.
    
    This function looks for:
    - Japanese quotes: 「text」 or 『text』
    - English quotes: "text" or 'text'
    - Lines after "Dialogue:" marker
    """
    dialogue_lines = []
    
    # Pattern 1: Japanese quotes 「text」 or 『text』
    jp_quotes = re.findall(r'[「『]([^」』]+)[」』]', text)
    dialogue_lines.extend(jp_quotes)
    
    # Pattern 2: English quotes "text" or 'text'
    en_quotes = re.findall(r'["\']([^"\']+)["\']', text)
    dialogue_lines.extend(en_quotes)
    
    # Pattern 3: Lines with "Dialogue:" prefix
    lines = text.split('\n')
    for line in lines:
        if 'Dialogue:' in line:
            # Extract text after "Dialogue:"
            match = re.search(r'Dialogue:\s*(.+)', line)
            if match:
                dialogue_lines.append(match.group(1))
    
    return dialogue_lines


def clean_text(text: str) -> str:
    """
    Clean extracted text by removing unwanted patterns.
    
    Removes:
    - Square brackets: [command]
    - Control codes: \code123
    - Extra whitespace
    """
    # Remove game commands in brackets [like this]
    text = re.sub(r'\[.*?\]', '', text)
    
    # Remove control codes \code123
    text = re.sub(r'\\[a-z]+\d*', '', text)
    
    # Normalize whitespace (multiple spaces -> single space)
    text = re.sub(r'\s+', ' ', text)
    
    # Remove leading/trailing whitespace
    text = text.strip()
    
    return text


def process_file(input_path: str, output_path: str):
    """
    Complete file processing pipeline.
    
    Steps:
    1. Read the file
    2. Extract dialogue
    3. Clean each line
    4. Write to output file
    """
    print(f"Processing: {input_path}")
    
    try:
        # Step 1: Read file
        content = read_file(input_path)
        print(f"  File size: {len(content)} characters")
        
        # Step 2: Extract dialogue
        dialogue = extract_dialogue(content)
        print(f"  Found {len(dialogue)} dialogue items")
        
        # Step 3: Clean each line
        cleaned = []
        for item in dialogue:
            cleaned_item = clean_text(item)
            if cleaned_item:  # Only add non-empty items
                cleaned.append(cleaned_item)
        
        # Step 4: Write output
        output_path_obj = Path(output_path)
        output_path_obj.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for item in cleaned:
                f.write(item + '\n')
        
        print(f"  ✓ Extracted {len(cleaned)} cleaned lines")
        print(f"  ✓ Saved to: {output_path}")
        
    except Exception as e:
        print(f"  ✗ Error: {e}")


def main():
    """
    Main function - handles command-line arguments.
    """
    # Check for command-line argument
    if len(sys.argv) < 2:
        print("Usage: python example_scraper.py <input_file>")
        print("\nExample:")
        print("  python example_scraper.py input.txt")
        print("  python example_scraper.py data/dialogue.txt")
        sys.exit(1)
    
    # Get input and output paths
    input_file = sys.argv[1]
    input_path = Path(input_file)
    
    # Check if input file exists
    if not input_path.exists():
        print(f"Error: File not found: {input_file}")
        sys.exit(1)
    
    # Create output path
    output_file = f"output/cleaned_{input_path.name}"
    
    # Process the file
    process_file(str(input_path), output_file)
    
    print("\nDone!")


if __name__ == '__main__':
    main()

