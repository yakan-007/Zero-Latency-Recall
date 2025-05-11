# Zero-Latency Recall (ZLR)

日本語 PDF を高速かつ高精度にテキスト抽出するオープンソース・ユーティリティです。縦書き・横書き混在や複数カラムなど、実務で遭遇する多様なレイアウトを想定しています。

---

## 特長

* **ゼロレイテンシ動作**: 依存ライブラリが揃っていれば即座に抽出が完了します。
* **2 モード構成**  
  * **Simple (Standard) モード** … 追加ライブラリ不要。`pdfplumber` ＋簡易ヒューリスティックで軽快。  
  * **Advanced (Pro) モード** … `layoutparser`・`paddle-ocr`・`pytesseract` を組み合わせた高精度抽出 (任意)。
* **マルチカラム対応**: PyMuPDF (`fitz`) のブロック情報を用いた列推定アルゴリズムを実装。
* **縦書き判定 & OCR フォールバック** (Pro): `pytesseract` の `jpn_vert` を優先し、必要に応じて画像を回転して再 OCR。
* **SQLite 出力**: 段落単位で正規化し `docs` テーブルへ保存、全文検索にもそのまま利用可能。
* **pytest テストスイート**: Ground Truth との文字列類似度で精度をスコアリング。

---

## デモ

```
# Simple モード (依存は requirements.txt のみ)
$ python extract.py tests/sample_pdfs/01_single-column_basic-layout.pdf \
        --db_path extracted.sqlite --edition free

# Advanced モード (追加依存が整っていれば自動で切替)
$ python extract.py tests/sample_pdfs/08_multi-column_magazine-style.pdf \
        --db_path extracted.sqlite --edition pro --force_ocr
```

抽出結果は `docs` テーブルにページ・段落番号付きで保存されます。

---

## インストール

```
$ git clone https://github.com/yourname/zlr-dev.git
$ cd zlr-dev
$ python -m venv .venv && source .venv/bin/activate
$ pip install --upgrade pip
$ pip install -r requirements.txt

# Pro モードを使う場合 (任意)
$ pip install layoutparser[layoutmodels] paddleocr pytesseract
```

**Python 3.9 以上**を推奨します。macOS／Linux で動作確認済み。

---

## ディレクトリ構成 (抜粋)

```
extract.py              # エントリポイント (CLI)
extractor/              # Simple / Advanced 抽出器実装
  ├─ simple.py
  ├─ advanced.py
  └─ base.py
tests/
  ├─ sample_pdfs/       # テスト用 PDF (追跡対象)
  ├─ ground_truth/      # 期待テキスト
  └─ test_extraction_quality.py
README.md               # 本ファイル
requirements.txt        # 最低限依存
```

---

## 使い方

### 基本

```
$ python extract.py <PDF ファイル> [--db_path OUTPUT.sqlite] [--edition free|pro] [--patent]
```

* `--edition` … `free` (デフォルト) か `pro` を指定。
* `--patent` … 特許公報用パーサを使用します。

### CLI スクリプト

簡易 CLI ラッパー `cli.py` も用意しています。

```
$ python cli.py -i path/to/dir_with_pdfs -o output.sqlite --edition free
```

ディレクトリを再帰走査して一括抽出します。

---

## テスト

Ground Truth 付きの回帰テストを実装しています。

```
$ pytest -q tests/test_extraction_quality.py
```

類似度が閾値を下回ると詳細な差分が表示されます。新しい PDF を追加する場合は、対応するテキストファイルを `tests/ground_truth/` に配置してください。

---

## 貢献

* Issue や Pull Request は日本語 / English どちらでも歓迎します。
* 新しいレイアウトへの対応・速度最適化・OCR 精度向上などのアイデアをお待ちしています。

### コーディング規約

* `black` + `isort` + `flake8` で整形 / lint 済み。
* 型ヒント (PEP 484) をできる限り付与しています。

---

## ライセンス

MIT License

---

## 謝辞

本プロジェクトは以下 OSS に依存しています。

* [pdfplumber](https://github.com/jsvine/pdfplumber)
* [PyMuPDF](https://github.com/pymupdf/PyMuPDF)
* [pytesseract](https://github.com/madmaze/pytesseract)
* [paddleocr](https://github.com/PaddlePaddle/PaddleOCR)
* [LayoutParser](https://github.com/Layout-Parser/layout-parser)

---

開発・運用に関するご意見やご要望は Issue でお気軽にお知らせください。 