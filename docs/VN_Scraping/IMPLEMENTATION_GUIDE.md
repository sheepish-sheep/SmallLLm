# Python File Scraper - Step-by-Step Implementation Guide

This guide walks you through implementing a Python file scraper from scratch.

---

## Tools & Libraries Used in Python Scraping

### **Core Python Libraries (Built-in)**
1. **`os`** - Operating system interface (file paths, directory operations)
2. **`pathlib`** - Modern file path handling (Path objects, easier than os.path)
3. **`re`** - Regular expressions (pattern matching, text extraction)
4. **`json`** - Parse JSON files
5. **`csv`** - Parse CSV files
6. **`xml.etree.ElementTree`** - Parse XML files
7. **`struct`** - Parse binary data structures
8. **`io`** - String/file I/O operations
9. **`sys`** - Command-line arguments, exit codes

### **External Libraries (Install via pip)**
1. **`chardet`** - Detect file encoding automatically
   ```bash
   pip install chardet
   ```

2. **`beautifulsoup4`** - Parse HTML/XML (if scraping HTML files)
   ```bash
   pip install beautifulsoup4
   ```

3. **`lxml`** - Fast XML/HTML parser
   ```bash
   pip install lxml
   ```

4. **`pandas`** - Data processing (for CSV/Excel files)
   ```bash
   pip install pandas
   ```

5. **`openpyxl`** - Read Excel files (.xlsx)
   ```bash
   pip install openpyxl
   ```

---

## Step-by-Step Implementation

### **STEP 1: Set Up Your Environment**

1. **Create your scraper file:**
   ```bash
   # Copy the template
   cp scraper_template.py my_scraper.py
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python -m venv venv
   venv\Scripts\activate  # Windows
   # or
   source venv/bin/activate  # Linux/Mac
   ```

3. **Install dependencies:**
   ```bash
   pip install chardet  # If needed for encoding detection
   ```

---

### **STEP 2: Implement File Reading (Function: `read_file`)**

**Goal:** Read files with proper encoding handling.

**Implementation:**

```python
def read_file(file_path: str, encoding: str = 'utf-8') -> str:
    """Read a text file and return its contents."""
    try:
        with open(file_path, 'r', encoding=encoding) as f:
            return f.read()
    except UnicodeDecodeError:
        # Try different encodings if default fails
        encodings = ['utf-8', 'shift_jis', 'latin-1', 'cp1252']
        for enc in encodings:
            try:
                with open(file_path, 'r', encoding=enc) as f:
                    return f.read()
            except (UnicodeDecodeError, UnicodeError):
                continue
        raise ValueError(f"Could not decode {file_path} with any encoding")
```

**For binary files:**

```python
def read_binary_file(file_path: str) -> bytes:
    """Read a binary file and return raw bytes."""
    with open(file_path, 'rb') as f:
        return f.read()
```

**Key Points:**
- Use `encoding` parameter to specify file encoding
- Handle `UnicodeDecodeError` by trying multiple encodings
- Use `'rb'` mode for binary files

---

### **STEP 3: Implement Data Extraction (Function: `extract_data`)**

**Goal:** Extract relevant data from file contents using patterns.

**Implementation Examples:**

**A) Extract text in quotes:**
```python
def extract_data(text: str) -> List[str]:
    """Extract dialogue in quotes."""
    dialogue_lines = []
    
    # Japanese quotes: 「text」 or 『text』
    jp_quotes = re.findall(r'[「『]([^」』]+)[」』]', text)
    dialogue_lines.extend(jp_quotes)
    
    # English quotes
    en_quotes = re.findall(r'["\']([^"\']+)["\']', text)
    dialogue_lines.extend(en_quotes)
    
    return dialogue_lines
```

**B) Extract lines matching a pattern:**
```python
def extract_data(text: str) -> List[str]:
    """Extract lines containing specific patterns."""
    lines = text.split('\n')
    extracted = []
    
    for line in lines:
        # Example: Extract lines with "Dialogue:" prefix
        if line.startswith('Dialogue:'):
            extracted.append(line.replace('Dialogue:', '').strip())
        
        # Example: Extract lines with Japanese characters
        if re.search(r'[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FAF]', line):
            extracted.append(line.strip())
    
    return extracted
```

**C) Extract using regex groups:**
```python
def extract_with_regex(text: str, pattern: str) -> List[str]:
    """Extract data using regex pattern."""
    matches = re.findall(pattern, text, re.MULTILINE | re.DOTALL)
    return matches

# Usage:
# dialogue = extract_with_regex(text, r'Character:\s*(.+)')
```

**Key Points:**
- Use `re.findall()` to find all matches
- Use `re.search()` to find first match
- Use `re.compile()` for repeated patterns (faster)
- Test patterns with https://regex101.com/

---

### **STEP 4: Implement Data Cleaning (Function: `clean_text`)**

**Goal:** Remove unwanted characters, normalize text.

**Implementation:**

```python
def clean_text(text: str) -> str:
    """Clean extracted text."""
    # Remove game commands in brackets [like this]
    text = re.sub(r'\[.*?\]', '', text)
    
    # Remove control codes \code123
    text = re.sub(r'\\[a-z]+\d*', '', text)
    
    # Remove special characters (keep letters, numbers, spaces, basic punctuation)
    text = re.sub(r'[^\w\s\.,!?;:\-「」『』]', '', text)
    
    # Normalize whitespace (multiple spaces -> single space)
    text = re.sub(r'\s+', ' ', text)
    
    # Remove leading/trailing whitespace
    text = text.strip()
    
    return text
```

**Key Points:**
- Use `re.sub()` to replace patterns
- Normalize whitespace for consistency
- Remove control codes and metadata
- Keep only relevant characters

---

### **STEP 5: Implement File Processing Pipeline (Function: `process_file`)**

**Goal:** Combine reading, extraction, and cleaning into one function.

**Implementation:**

```python
def process_file(input_path: str, output_path: str) -> None:
    """Process a single file: read, extract, clean, and save."""
    print(f"Processing: {input_path}")
    
    # Step 1: Read the file
    try:
        content = read_file(input_path)
    except Exception as e:
        print(f"  Error reading file: {e}")
        return
    
    # Step 2: Extract relevant data
    extracted_data = extract_data(content)
    
    # Step 3: Clean each extracted item
    cleaned_data = []
    for item in extracted_data:
        cleaned = clean_text(item)
        if cleaned:  # Only add non-empty items
            cleaned_data.append(cleaned)
    
    # Step 4: Write to output file
    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)  # Create dirs if needed
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in cleaned_data:
            f.write(item + '\n')
    
    print(f"  Extracted {len(cleaned_data)} items")
```

**Key Points:**
- Handle errors gracefully with try/except
- Create output directory if it doesn't exist
- Always write output as UTF-8
- Print progress messages

---

### **STEP 6: Implement Main Function**

**Goal:** Handle command-line arguments and process files.

**Implementation:**

```python
def main():
    """Main entry point for the scraper."""
    import sys
    
    # Step 1: Get input path from command line or use default
    if len(sys.argv) > 1:
        input_path = Path(sys.argv[1])
    else:
        input_path = Path('input_file.txt')  # Default
    
    # Step 2: Set up output directory
    output_dir = Path('output')
    output_dir.mkdir(exist_ok=True)
    
    # Step 3: Process file(s)
    if input_path.is_file():
        # Single file
        output_file = output_dir / f'cleaned_{input_path.name}'
        process_file(str(input_path), str(output_file))
    
    elif input_path.is_dir():
        # Directory of files
        for file_path in input_path.rglob('*'):
            if file_path.is_file():
                rel_path = file_path.relative_to(input_path)
                output_file = output_dir / rel_path.with_suffix('.txt')
                output_file.parent.mkdir(parents=True, exist_ok=True)
                process_file(str(file_path), str(output_file))
    
    else:
        print(f"Error: {input_path} not found")
        print("\nUsage:")
        print("  python my_scraper.py input_file.txt")
        print("  python my_scraper.py input_directory/")


if __name__ == '__main__':
    main()
```

**Key Points:**
- Use `sys.argv` for command-line arguments
- Handle both single files and directories
- Provide usage instructions on error
- Use `Path.rglob('*')` to recursively find files

---

## Complete Example Implementation

Here's a minimal working example:

```python
import re
from pathlib import Path
import sys

def read_file(file_path: str) -> str:
    """Read file with encoding fallback."""
    encodings = ['utf-8', 'shift_jis', 'latin-1']
    for enc in encodings:
        try:
            with open(file_path, 'r', encoding=enc) as f:
                return f.read()
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Could not decode {file_path}")

def extract_dialogue(text: str) -> list:
    """Extract dialogue in quotes."""
    dialogue = []
    # Find Japanese quotes
    dialogue.extend(re.findall(r'[「『]([^」』]+)[」』]', text))
    # Find English quotes
    dialogue.extend(re.findall(r'["\']([^"\']+)["\']', text))
    return dialogue

def clean_text(text: str) -> str:
    """Clean text."""
    text = re.sub(r'\[.*?\]', '', text)  # Remove brackets
    text = re.sub(r'\s+', ' ', text)     # Normalize spaces
    return text.strip()

def process_file(input_path: str, output_path: str):
    """Process file pipeline."""
    content = read_file(input_path)
    dialogue = extract_dialogue(content)
    cleaned = [clean_text(d) for d in dialogue if clean_text(d)]
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(cleaned))
    print(f"Extracted {len(cleaned)} lines")

if __name__ == '__main__':
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
        output_file = f'output/cleaned_{Path(input_file).name}'
        process_file(input_file, output_file)
    else:
        print("Usage: python scraper.py input_file.txt")
```

---

## Testing Your Scraper

1. **Test on a small file first:**
   ```bash
   python my_scraper.py test_file.txt
   ```

2. **Check the output:**
   ```bash
   # View output
   type output\cleaned_test_file.txt  # Windows
   cat output/cleaned_test_file.txt   # Linux/Mac
   ```

3. **Debug issues:**
   - Add `print()` statements to see what's being extracted
   - Test regex patterns separately
   - Check file encoding if you get decode errors

---

## Next Steps

1. **Customize extraction patterns** for your specific file format
2. **Add error handling** for edge cases
3. **Add logging** instead of print statements
4. **Add command-line arguments** using `argparse`
5. **Add progress bars** using `tqdm` library
6. **Add unit tests** to verify functionality

---

## Common Patterns

### Extract lines between markers:
```python
pattern = r'START_MARKER(.*?)END_MARKER'
matches = re.findall(pattern, text, re.DOTALL)
```

### Extract key-value pairs:
```python
pattern = r'(\w+):\s*(.+)'
pairs = re.findall(pattern, text)
```

### Skip empty lines:
```python
lines = [line.strip() for line in text.split('\n') if line.strip()]
```

Good luck with your scraper!

