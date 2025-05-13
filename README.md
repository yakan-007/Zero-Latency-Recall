# Zero-Latency Recall (ZLR)

ZLR (Zero-Latency Recall) is an open-source utility designed to extract information from PDF files in a local environment and **instantly (with zero latency) recall relevant information in response to search queries.** It focuses on efficiently processing Japanese PDFs with complex layouts (mixed vertical/horizontal text, multiple columns) and integrating with knowledge bases like Obsidian.

**In this project, "zero latency" primarily refers to the search response speed after the database has been built. The information extraction (ingestion) process from PDFs may take time depending on the file size and complexity.**

ZLR aims to be a powerful "second brain" for anyone who uses PDFs as an information source.

## Project Scope and Goals

ZLR provides two main functionalities:

1.  **High-Accuracy PDF Extraction Toolkit**:
    *   Text extraction capabilitiesรองรับ diverse Japanese PDF layouts.
    *   Structured storage of extracted text and metadata (filename, page number, tags, etc.) in an SQLite database.
2.  **Local Full-Text Search and Obsidian Integration**:
    *   Fast full-text search leveraging SQLite FTS5.
    *   Automatic generation of Obsidian notes for each extracted paragraph, enabling seamless knowledge base integration.

Currently, the user interface is primarily command-line based (CLI), with future plans for a richer GUI and API.

---

## Features

*   **Fast Text Extraction**: Utilizes `pdfplumber` and `PyMuPDF` (`fitz`) for efficient text extraction.
*   **Flexible Extraction Strategies**:
    *   `SimpleExtractor`: Basic extraction based on `pdfplumber`.
    *   `AdvancedExtractor`: High-accuracy extraction combining OCR (`pytesseract`) and `PyMuPDF` (implementation is extensible).
*   **Multi-Column Support**: Analyzes column layouts using multiple heuristics based on word information from `pdfplumber` or block information from `PyMuPDF`.
*   **Vertical Text Support**: Detects vertical text layouts and uses `pdfplumber`'s or `PyMuPDF`'s vertical mode, or OCR (`jpn_vert`).
*   **SQLite Output**: Normalizes extracted results paragraph by paragraph and saves them to a `docs` (FTS5) table, ready for full-text search.
*   **Obsidian Integration**: Extracted paragraphs are linked with the search function and can be automatically integrated into an Obsidian knowledge base.
*   **Folder Watch Feature**: Automatically processes PDF files added to a specified folder.
*   **Modular Design**: Extraction logic, database operations, and CLI are separated for improved extensibility and maintainability.
*   **pytest Test Suite**: Evaluates extraction accuracy by string similarity with ground truth.

---

## Demo

```bash
# Extract using SimpleExtractor (only requires dependencies in requirements.txt)
python -m extract tests/sample_pdfs/01_single-column_basic-layout.pdf \\
        --db_path extracted.sqlite --edition free

# Extract using AdvancedExtractor (requires additional dependencies, planned for future implementation)
# python -m extract tests/sample_pdfs/08_multi-column_magazine-style.pdf \\
#         --db_path extracted.sqlite --edition pro --force_ocr
```

The extracted results are saved in the `docs` table of the SQLite file specified by `--db_path`, along with filename, page, and paragraph number.

---

## Installation

```bash
git clone https://github.com/your-username/zlr-dev.git # Replace with your repository URL
cd zlr-dev
python -m venv .venv
source .venv/bin/activate  # macOS / Linux
# .venv\\Scripts\\activate  # Windows

pip install --upgrade pip
pip install -r requirements.txt

# (Optional) For advanced features (OCR, PyMuPDF)
pip install pytesseract Pillow pdf2image PyMuPDF opencv-python numpy

# Tesseract OCR and Poppler also need to be installed separately
# macOS: brew install tesseract tesseract-lang poppler
# Ubuntu: sudo apt update && sudo apt install -y tesseract-ocr tesseract-ocr-jpn tesseract-ocr-jpn-vert poppler-utils
# Windows: Use installers and add to PATH
```

**Python 3.9 or higher** is recommended.

---

## Directory Structure (Key Parts)

```
.
├── extract/                  # Main PDF extraction package
│   ├── __main__.py           # CLI entry point
│   ├── core/                 # Core extraction logic
│   │   ├── extractor.py      # Extraction functions and helpers
│   │   └── strategies.py     # BaseExtractor, SimpleExtractor, AdvancedExtractor
│   ├── db/
│   │   └── repo.py           # SQLite database operations
│   ├── config.py             # Configuration file (DB path, etc.)
│   ├── obsidian_export.py    # Obsidian integration utility
│   └── tags_JP.yaml          # Keyword dictionary for paragraph tagging
├── tests/                    # Test code
│   ├── sample_pdfs/          # Test PDFs
│   ├── ground_truth/         # Expected text output
│   ├── test_accuracy.py      # Accuracy tests (Simple/Advanced)
│   └── test_extraction_quality.py # Extraction quality tests (multi-column, etc.)
├── search.py                 # Search script (Obsidian integration, etc.)
├── zlr_watch.py              # Folder watching script
├── README.md                 # This file
└── requirements.txt          # Python dependencies
```

---

## Usage

### Extraction

```bash
python -m extract <PDF_files...> [--db_path <output_DB_path>] [--edition <free|pro>] [--patent] [--force_ocr] [--log-level <LEVEL>]
```

*   `<PDF_files...>`: One or more PDF files to process.
*   `--db_path`: Output SQLite file path (default: `zlr.sqlite`).
*   `--edition`: Specify `free` (default) or `pro`. `pro` is currently equivalent to Simple.
*   `--patent`: Enable extraction settings optimized for patent documents.
*   `--force_ocr`: (When `pro` is specified) Force OCR on all pages.
*   `--log-level`: Logging verbosity (DEBUG, INFO, WARNING, ERROR, CRITICAL).

### Folder Watch

Monitors a specified folder and automatically runs the extraction process when new PDF files are added.

```bash
# Set environment variables (if needed)
export OBSIDIAN_VAULT=/path/to/your/obsidian/vault  # For Obsidian integration

# Start watching
python zlr_watch.py
```

The watch folder is specified by `WATCH_FOLDER_PATH` in `extract/config.py`. Processing logs are displayed in the terminal during monitoring. Press `Ctrl+C` to stop.

### Search

Searches and displays content saved in the database.

```bash
python search.py <keywords...> [-t <tags>] [-l <limit>] [-o] [--db_path <DB_path>]
```

*   `<keywords...>`: Keywords to search for (multiple keywords perform an AND search).
*   `-t, --tags`: Search only paragraphs with the specified tags (comma-separated).
*   `-l, --limit`: Maximum number of results to display (default: 50).
*   `-o, --obsidian`: Output results in Markdown format for Obsidian integration.
*   `--db_path`: SQLite file path to search (default: `zlr.sqlite`).

### Obsidian Integration

ZLR integrates deeply with Obsidian to incorporate PDF content into your knowledge management system.

**Setup**:
1. Set the `OBSIDIAN_VAULT` environment variable to your Obsidian Vault path:
   ```bash
   # macOS / Linux
   export OBSIDIAN_VAULT=/path/to/your/obsidian/vault
   
   # Windows
   set OBSIDIAN_VAULT=C:\\path\\to\\your\\obsidian\\vault
   ```
   
   For persistence, create a `.env` file in the project root:
   ```
   OBSIDIAN_VAULT=/path/to/your/obsidian/vault
   ```

**Integration Features**:
1. **Paragraph Snippets**: During PDF extraction, each paragraph is automatically saved as an individual Markdown file.
   - Destination: `<OBSIDIAN_VAULT>/zlr-inbox/zlr-snippets/`
   - File Format: YAML Front Matter + block quote
   - Filename: `<doc_slug>_p<page>_<idx>.md`

2. **Search Results**: When using the `-o` option during a search, results are saved in Markdown format and include links to each snippet.
   - Destination: `<OBSIDIAN_VAULT>/zlr-inbox/`

This mechanism allows you to leverage Obsidian's powerful features (backlinks, tags, search, Dataview, etc.) to build your knowledge base.

---

## Basic Workflow

1. **Environment Setup**:
   ```bash
   export OBSIDIAN_VAULT=/path/to/your/obsidian/vault  # For Obsidian integration
   ```

2. **PDF Extraction**:
   ```bash
   python -m extract sample.pdf
   ```
   Or automatic processing with watch mode:
   ```bash
   python zlr_watch.py
   ```

3. **Search and Reference**:
   ```bash
   python search.py keyword -o
   ```
   Check the `zlr-inbox` folder in Obsidian to find the search results and snippets.

4. **Knowledge Utilization**: Reference PDF content in Obsidian while citing and linking to your notes.

---

## Testing

Runs regression tests based on similarity with Ground Truth text.

```bash
pytest
```

You can also specify detailed test cases.

```bash
pytest tests/test_extraction_quality.py::test_multi_column_accuracy
```

---

## Contributing

*   Issues and Pull Requests are welcome in either Japanese or English.
*   We welcome proposals for supporting new layouts, improving extraction accuracy, and enhancing performance.

### Coding Conventions

*   Format and lint with `ruff` (equivalent to `black`, `isort`, `flake8`).
*   Use type hints (PEP 484) as much as possible.

---

## License

MIT License

---

## Acknowledgments

This project relies on excellent open-source software such as:

*   [pdfplumber](https://github.com/jsvine/pdfplumber)
*   [PyMuPDF (fitz)](https://github.com/pymupdf/PyMuPDF)
*   [pytesseract](https://github.com/madmaze/pytesseract) & [Tesseract OCR](https://github.com/tesseract-ocr/tesseract)
*   [pdf2image](https://github.com/Belval/pdf2image) & [Poppler](https://poppler.freedesktop.org/)
*   [PyYAML](https://pyyaml.org/)
*   [pytest](https://docs.pytest.org/)

---

Please feel free to share your opinions and requests regarding development and operation via Issues. 