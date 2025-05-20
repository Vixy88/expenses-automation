# Expense Automation Tool

This tool automatically processes expense files (PDF, JPG, PNG) to extract:

- Company/vendor name
- Cost and currency
- Date of transaction
- Auto-categorization of expenses

It converts all amounts to EURO and creates a CSV report.

## Features

- Processes multiple file formats (PDF, JPG, JPEG, PNG)
- Uses OCR to extract text from images and scanned documents
- Automatically extracts key information from invoices and receipts
- Converts different currencies (USD, GBP, EUR) to EURO
- Automatically categorizes expenses based on keywords
- Produces a detailed CSV report and category summary
- Tracks unprocessed files with partial data extraction
- Optional OpenAI integration for improved extraction (requires API key)

## Installation

1. Clone this repository or download the script

2. Install required Python packages:

```bash
pip install pandas requests pillow pytesseract pdf2image PyPDF2
```

3. Install Tesseract OCR (required for image processing):

- macOS:

```bash
brew install tesseract
```

- Ubuntu/Debian:

```bash
sudo apt-get install tesseract-ocr
```

- Windows: Download installer from https://github.com/UB-Mannheim/tesseract/wiki

4. Install Poppler (required for PDF processing):

- macOS:

```bash
brew install poppler
```

- Ubuntu/Debian:

```bash
sudo apt-get install poppler-utils
```

- Windows: Download and install from http://blog.alivate.com.au/poppler-windows/

5. (Optional) Install OpenAI Python package for AI-assisted extraction:

```bash
pip install openai
```

## Usage

Run the script with the folder containing your expense files:

```bash
python expenses_automation.py /path/to/expense/folder
```

Optional arguments:

- `--output` or `-o`: Specify output CSV file name (default: expenses.csv)
- `--use-openai`: Use OpenAI API to help extract data from difficult documents
- `--openai-api-key`: Provide your OpenAI API key (or set as OPENAI_API_KEY environment variable)
- `--portugal-tax`: Optimize categorization for Portuguese freelancer taxes
- `--recursive`: Search for expense files recursively in subdirectories

Example:

```bash
# Basic usage
python expenses_automation.py ~/Documents/Expenses --output 2023_expenses.csv

# With OpenAI integration
python expenses_automation.py ~/Documents/Expenses --use-openai --openai-api-key=sk-your-key-here

# With Portuguese tax optimization and recursive search
python expenses_automation.py ~/Documents/Expenses --portugal-tax --recursive
```

## Output

The tool generates three CSV files:

1. Main CSV report with all expenses (default: expenses.csv)
2. Summary CSV report with totals by category (default: expenses_summary.csv)
3. Unprocessed files report with partial data and reasons (default: expenses_unprocessed.csv)

## Handling Unprocessed Expenses

For files that couldn't be fully processed, check the `expenses_unprocessed.csv` file, which contains:

- Filename
- Any partial data that was extracted (vendor, amount, date)
- Reason for failure
- Excerpt of extracted text

You can:

1. Manually review these files
2. Try using the OpenAI integration to improve extraction
3. Manually add these expenses to the main CSV

## Limitations

- OCR accuracy depends on the quality of the scanned documents/images
- Currency detection might need manual verification for unusual formats
- Some invoices with complex layouts may not be parsed correctly
