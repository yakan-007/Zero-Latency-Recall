# Zero-Latency Recall (ZLR)

日本語 PDF を高速かつ高精度にテキスト抽出するオープンソース・ユーティリティです。縦書き・横書き混在や複数カラムなど、実務で遭遇する多様なレイアウトを想定しています。

---

## 特長

*   **高速なテキスト抽出**: `pdfplumber` や `PyMuPDF` (`fitz`) を利用し、効率的なテキスト抽出を実現。
*   **柔軟な抽出戦略**:
    *   `SimpleExtractor`: `pdfplumber` ベースの基本的な抽出。
    *   `AdvancedExtractor`: OCR (`pytesseract`) や `PyMuPDF` を組み合わせた高精度抽出（実装は拡張可能）。
*   **マルチカラム対応**: `pdfplumber` の単語情報や `PyMuPDF` のブロック情報を用いた複数のヒューリスティックで段組みを解析。
*   **縦書き対応**: 縦書きレイアウトを判定し、`pdfplumber` や `PyMuPDF` の縦書きモード、または OCR (`jpn_vert`) を利用。
*   **SQLite 出力**: 抽出結果を段落単位で正規化し、`docs` (FTS5) テーブルへ保存。全文検索にそのまま利用可能。
*   **モジュール化された設計**: 抽出ロジック、DB操作、CLIが分離され、拡張性とメンテナンス性が向上。
*   **pytest テストスイート**: Ground Truth との文字列類似度で抽出精度を評価。

---

## デモ

```bash
# SimpleExtractor を使用して抽出 (依存は requirements.txt のみ)
python -m extract tests/sample_pdfs/01_single-column_basic-layout.pdf \
        --db_path extracted.sqlite --edition free

# AdvancedExtractor を使用 (追加依存が必要、将来的に実装予定)
# python -m extract tests/sample_pdfs/08_multi-column_magazine-style.pdf \
#         --db_path extracted.sqlite --edition pro --force_ocr
```

抽出結果は `--db_path` で指定した SQLite ファイルの `docs` テーブルに、ファイル名・ページ・段落番号付きで保存されます。

---

## インストール

```bash
git clone https://github.com/your-username/zlr-dev.git # あなたのリポジトリURLに書き換えてください
cd zlr-dev
python -m venv .venv
source .venv/bin/activate  # macOS / Linux
# .venv\Scripts\activate  # Windows

pip install --upgrade pip
pip install -r requirements.txt

# (任意) 高度な機能 (OCR, PyMuPDF) を利用する場合
pip install pytesseract Pillow pdf2image PyMuPDF opencv-python numpy

# Tesseract OCR と Poppler のインストールも別途必要です
# macOS: brew install tesseract tesseract-lang poppler
# Ubuntu: sudo apt update && sudo apt install -y tesseract-ocr tesseract-ocr-jpn tesseract-ocr-jpn-vert poppler-utils
# Windows: インストーラを使用し、PATH を通してください
```

**Python 3.9 以上**を推奨します。

---

## ディレクトリ構成 (主要部)

```
.
├── extract/                  # PDF抽出メインパッケージ
│   ├── __main__.py           # CLI エントリポイント
│   ├── core/                 # 抽出コアロジック
│   │   ├── extractor.py      # 抽出関数・ヘルパー群
│   │   └── strategies.py     # BaseExtractor, SimpleExtractor, AdvancedExtractor
│   ├── db/
│   │   └── repo.py           # SQLite データベース操作
│   ├── config.py             # 設定ファイル (DBパスなど)
│   └── tags_JP.yaml          # 段落タグ付け用キーワード辞書
├── tests/                    # テストコード
│   ├── sample_pdfs/          # テスト用 PDF
│   ├── ground_truth/         # 期待するテキスト出力
│   ├── test_accuracy.py      # 精度テスト (Simple/Advanced)
│   └── test_extraction_quality.py # 抽出品質テスト (マルチカラムなど)
├── search.py                 # 検索スクリプト (Obsidian連携など)
├── README.md                 # 本ファイル
└── requirements.txt          # Python 依存パッケージ
```

---

## 使い方

### 抽出 (Extract)

```bash
python -m extract <PDFファイル...> [--db_path <出力DBパス>] [--edition <free|pro>] [--patent] [--force_ocr] [--log-level <LEVEL>]
```

*   `<PDFファイル...>`: 処理したい PDF ファイルを複数指定可能。
*   `--db_path`: 出力先の SQLite ファイルパス (デフォルト: `zlr.sqlite`)。
*   `--edition`: `free` (デフォルト) または `pro` を指定。`pro` は現在 Simple と同等。
*   `--patent`: 特許公報向けの抽出設定を有効化。
*   `--force_ocr`: (`pro` 指定時) 全てのページで強制的に OCR を実行。
*   `--log-level`: ログの詳細度 (DEBUG, INFO, WARNING, ERROR, CRITICAL)。

### 検索 (Search)

データベースに保存した内容を検索・表示します。

```bash
python search.py <キーワード...> [-t <タグ>] [-l <上限数>] [-o] [--db_path <DBパス>]
```

*   `<キーワード...>`: 検索したいキーワード (複数指定で AND 検索)。
*   `-t, --tags`: 指定したタグを持つ段落のみを検索 (カンマ区切り)。
*   `-l, --limit`: 表示する最大件数 (デフォルト: 50)。
*   `-o, --obsidian`: 結果を Obsidian 連携用の Markdown 形式で出力。
*   `--db_path`: 検索対象の SQLite ファイルパス (デフォルト: `zlr.sqlite`)。

---

## テスト

Ground Truth テキストとの類似度に基づいた回帰テストを実行します。

```bash
pytest
```

詳細なテストケースを指定することも可能です。

```bash
pytest tests/test_extraction_quality.py::test_multi_column_accuracy
```

---

## 貢献

*   Issue や Pull Request は日本語 / English どちらでも歓迎します。
*   新しいレイアウトへの対応、抽出精度の向上、パフォーマンス改善などの提案をお待ちしています。

### コーディング規約

*   `ruff` (`black`, `isort`, `flake8` 相当) でフォーマット・Lint を実施。
*   型ヒント (PEP 484) を可能な限り使用。

---

## ライセンス

MIT License

---

## 謝辞

本プロジェクトは以下のような優れたオープンソースソフトウェアに依存しています。

*   [pdfplumber](https://github.com/jsvine/pdfplumber)
*   [PyMuPDF (fitz)](https://github.com/pymupdf/PyMuPDF)
*   [pytesseract](https://github.com/madmaze/pytesseract) & [Tesseract OCR](https://github.com/tesseract-ocr/tesseract)
*   [pdf2image](https://github.com/Belval/pdf2image) & [Poppler](https://poppler.freedesktop.org/)
*   [PyYAML](https://pyyaml.org/)
*   [pytest](https://docs.pytest.org/)

---

開発・運用に関するご意見やご要望は Issue でお気軽にお知らせください。 