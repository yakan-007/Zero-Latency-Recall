"""ZLR 抽出 CLI

フェーズ 1: SimpleExtractor でのみ動作し、将来 AdvancedExtractor に切り替えられる設計。
使用例:

    python cli.py --pdf  sample.pdf --edition free
"""

from __future__ import annotations

import argparse
from pathlib import Path

from extractor import SimpleExtractor, AdvancedExtractor


def main() -> None:  # noqa: D401
    parser = argparse.ArgumentParser(description="PDF を抽出し段落を表示する簡易 CLI")
    parser.add_argument("pdf", help="入力 PDF")
    parser.add_argument("--edition", choices=["free", "pro"], default="free", help="使用するエディション (free=Simple, pro=Advanced)")
    parser.add_argument("--force_ocr", action="store_true", help="全ページ OCR を強制")
    args = parser.parse_args()

    pdf_path = Path(args.pdf)
    if not pdf_path.exists():
        raise SystemExit(f"PDF が見つかりません: {pdf_path}")

    if args.edition == "pro":
        extractor = AdvancedExtractor()
    else:
        extractor = SimpleExtractor()

    paragraphs = extractor.process_pdf(pdf_path, force_ocr=args.force_ocr)

    for fn, page, idx, text, source, tags in paragraphs:
        print("-" * 40)
        print(f"{fn} page={page} para={idx} source={source} tags={tags}")
        print(text[:200].replace("\n", " ") + ("…" if len(text) > 200 else ""))


if __name__ == "__main__":
    main() 