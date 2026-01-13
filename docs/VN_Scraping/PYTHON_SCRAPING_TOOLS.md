# Python Tools for File Scraping - Quick Reference

## 📚 Built-in Python Libraries (No Installation Needed)

### **File I/O Operations**

#### `open()` - Read/Write Files
```python
# Read text file
with open('file.txt', 'r', encoding='utf-8') as f:
    content = f.read()

# Read binary file
with open('file.dat', 'rb') as f:
    data = f.read()

# Write file
with open('output.txt', 'w', encoding='utf-8') as f:
    f.write('content')
```

#### `pathlib.Path` - Modern File Path Handling
```python
from pathlib import Path

# Create Path object
file_path = Path('data/file.txt')

# Check if exists
if file_path.exists():
    # Read file
    content = file_path.read_text(encoding='utf-8')

# Get all files in directory
files = list(Path('data').rglob('*.txt'))

# Create directory
output_dir = Path('output')
output_dir.mkdir(parents=True, exist_ok=True)
```

#### `os` - Operating System Interface
```python
import os

# Check if file exists
if os.path.exists('file.txt'):
    # Get file size
    size = os.path.getsize('file.txt')

# List directory contents
files = os.listdir('data/')

# Join paths
full_path = os.path.join('data', 'subfolder', 'file.txt')
```

---

### **Text Processing**

#### `re` - Regular Expressions
```python
import re

# Find all matches
matches = re.findall(r'pattern', text)

# Find first match
match = re.search(r'pattern', text)
if match:
    result = match.group(1)

# Replace patterns
cleaned = re.sub(r'\[.*?\]', '', text)

# Split by pattern
parts = re.split(r'[,;]', text)

# Compile pattern for reuse (faster)
pattern = re.compile(r'Character:\s*(.+)')
matches = pattern.findall(text)
```

#### `string` - String Operations
```python
import string

# Strip whitespace
text = text.strip()

# Split by delimiter
lines = text.split('\n')

# Join list into string
result = '\n'.join(lines)

# Check if contains substring
if 'dialogue' in text:
    # Do something
    pass
```

---

### **Data Parsing**

#### `json` - JSON Files
```python
import json

# Parse JSON file
with open('data.json', 'r') as f:
    data = json.load(f)

# Parse JSON string
data = json.loads(json_string)

# Write JSON
with open('output.json', 'w') as f:
    json.dump(data, f, indent=2)
```

#### `csv` - CSV Files
```python
import csv

# Read CSV
with open('data.csv', 'r', encoding='utf-8') as f:
    reader = csv.reader(f)
    for row in reader:
        print(row)

# Read CSV as dictionary
with open('data.csv', 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        print(row['column_name'])
```

#### `xml.etree.ElementTree` - XML Files
```python
import xml.etree.ElementTree as ET

# Parse XML
tree = ET.parse('data.xml')
root = tree.getroot()

# Find elements
for item in root.findall('item'):
    text = item.text
```

#### `struct` - Binary Data
```python
import struct

# Unpack binary data
# Format: 'I' = unsigned int, 'H' = unsigned short, 's' = string
data = struct.unpack('I', binary_data[:4])  # Read 4 bytes as int
```

---

### **Command Line & System**

#### `sys` - System-Specific Parameters
```python
import sys

# Get command-line arguments
args = sys.argv  # ['script.py', 'arg1', 'arg2']

# Exit program
sys.exit(0)  # Success
sys.exit(1)  # Error
```

#### `argparse` - Command-Line Arguments
```python
import argparse

parser = argparse.ArgumentParser(description='File scraper')
parser.add_argument('input', help='Input file or directory')
parser.add_argument('-o', '--output', default='output', help='Output directory')
args = parser.parse_args()
```

---

## 🔧 External Libraries (Install via pip)

### **Encoding Detection**

#### `chardet` - Auto-detect File Encoding
```python
import chardet

# Detect encoding
with open('file.txt', 'rb') as f:
    raw_data = f.read()
    result = chardet.detect(raw_data)
    encoding = result['encoding']

# Read with detected encoding
with open('file.txt', 'r', encoding=encoding) as f:
    content = f.read()
```

**Install:** `pip install chardet`

---

### **HTML/XML Parsing**

#### `beautifulsoup4` - Parse HTML/XML
```python
from bs4 import BeautifulSoup

# Parse HTML
with open('file.html', 'r') as f:
    soup = BeautifulSoup(f, 'html.parser')

# Find elements
dialogue = soup.find_all('div', class_='dialogue')

# Extract text
for item in dialogue:
    text = item.get_text()
```

**Install:** `pip install beautifulsoup4`

#### `lxml` - Fast XML/HTML Parser
```python
from lxml import etree

# Parse XML
tree = etree.parse('data.xml')

# XPath queries
items = tree.xpath('//item[@type="dialogue"]/text()')
```

**Install:** `pip install lxml`

---

### **Data Processing**

#### `pandas` - Data Analysis
```python
import pandas as pd

# Read CSV
df = pd.read_csv('data.csv')

# Read Excel
df = pd.read_excel('data.xlsx')

# Filter data
dialogue = df[df['type'] == 'dialogue']['text']

# Export
df.to_csv('output.csv', index=False)
```

**Install:** `pip install pandas`

#### `openpyxl` - Excel Files
```python
from openpyxl import load_workbook

# Load workbook
wb = load_workbook('data.xlsx')
ws = wb.active

# Read cells
for row in ws.iter_rows(values_only=True):
    print(row)
```

**Install:** `pip install openpyxl`

---

### **Progress & Logging**

#### `tqdm` - Progress Bars
```python
from tqdm import tqdm

# Progress bar for loops
for file in tqdm(files, desc="Processing"):
    process_file(file)
```

**Install:** `pip install tqdm`

#### `logging` - Better Logging (Built-in)
```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)
logger.info("Processing file...")
logger.error("Error occurred!")
```

---

## 🎯 Common Scraping Patterns

### Pattern 1: Extract Text Between Markers
```python
import re

# Extract text between START and END
pattern = r'START(.*?)END'
matches = re.findall(pattern, text, re.DOTALL)
```

### Pattern 2: Extract Key-Value Pairs
```python
# Format: "key: value"
pattern = r'(\w+):\s*(.+)'
pairs = re.findall(pattern, text)
```

### Pattern 3: Extract Quoted Text
```python
# Japanese quotes
jp_quotes = re.findall(r'[「『]([^」』]+)[」』]', text)

# English quotes
en_quotes = re.findall(r'["\']([^"\']+)["\']', text)
```

### Pattern 4: Skip Comments/Metadata
```python
# Remove lines starting with #
lines = [line for line in text.split('\n') 
         if not line.strip().startswith('#')]
```

### Pattern 5: Extract Structured Blocks
```python
# Extract blocks like [BLOCK]...content...[/BLOCK]
pattern = r'\[BLOCK\](.*?)\[/BLOCK\]'
blocks = re.findall(pattern, text, re.DOTALL)
```

---

## 📦 Recommended Requirements.txt

```txt
# Encoding detection
chardet>=5.0.0

# HTML/XML parsing (if needed)
beautifulsoup4>=4.12.0
lxml>=4.9.0

# Data processing (if needed)
pandas>=2.0.0
openpyxl>=3.1.0

# Progress bars (optional but nice)
tqdm>=4.65.0
```

---

## 🚀 Quick Start Example

```python
import re
from pathlib import Path

def scrape_file(input_file: str, output_file: str):
    """Simple file scraper example."""
    # Read file
    content = Path(input_file).read_text(encoding='utf-8')
    
    # Extract using regex
    pattern = r'Dialogue:\s*(.+)'
    matches = re.findall(pattern, content)
    
    # Clean results
    cleaned = [m.strip() for m in matches if m.strip()]
    
    # Write output
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    Path(output_file).write_text('\n'.join(cleaned), encoding='utf-8')
    
    print(f"Extracted {len(cleaned)} items")

if __name__ == '__main__':
    scrape_file('input.txt', 'output/cleaned.txt')
```

---

## 💡 Tips

1. **Always specify encoding** when reading files (UTF-8 is most common)
2. **Use `pathlib.Path`** instead of `os.path` for cleaner code
3. **Test regex patterns** on https://regex101.com/ before using
4. **Handle errors gracefully** with try/except blocks
5. **Process small files first** to test your logic
6. **Use context managers** (`with open()`) for file handling
7. **Create output directories** before writing files

