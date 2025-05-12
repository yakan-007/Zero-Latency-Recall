"""ZLR 抽出 CLI

フェーズ 1: SimpleExtractor でのみ動作し、将来 AdvancedExtractor に切り替えられる設計。
使用例:

    python -m extract --pdf  sample.pdf --edition free
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from .core.extractor import extract_and_process_pdf, AdvancedExtractor, SimpleExtractor
from .db.repo import ensure_db_schema, save_paragraphs_to_db
from .config import DB_PATH


def main() -> None:  # noqa: D401
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(module)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    logging.getLogger("pdfminer").setLevel(logging.WARNING)
    logging.getLogger("pdfplumber").setLevel(logging.WARNING)

    parser = argparse.ArgumentParser(description="PDF を抽出し SQLite(FTS5) に保存")
    parser.add_argument("pdf", nargs='+', help="入力 PDF ファイル")
    parser.add_argument("--db_path", type=Path, default=DB_PATH, help=f"出力 DB ファイル (デフォルト: {DB_PATH})")
    parser.add_argument("--patent", action="store_true", help="特許モード (pdfplumber 高密度レイアウト)")
    parser.add_argument(
        "--edition",
        choices=["free", "pro"],
        default="free",
        help="抽出エディションを選択 (free=従来 pdfplumber, pro=AdvancedExtractor)",
    )
    parser.add_argument(
        "--force_ocr",
        action="store_true",
        help="(Pro 限定) すべてのページを OCR で抽出する",
    )
    parser.add_argument(
        "--log-level",
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        default='INFO',
        help="ログレベルを設定 (デフォルト: INFO)"
    )

    args = parser.parse_args()

    log_level = getattr(logging, args.log_level.upper(), logging.INFO)
    logging.getLogger().setLevel(log_level)

    logging.info(f"データベースパス: {args.db_path}")
    logging.info(f"エディション: {args.edition}, 特許モード: {args.patent}, 強制OCR: {args.force_ocr}")
    logging.info(f"ログレベル: {args.log_level}")

    ensure_db_schema(args.db_path)

    advanced_extractor = None
    simple_extractor = None
    if args.edition == "pro":
        try:
            advanced_extractor = AdvancedExtractor(patent_mode=args.patent)
            logging.info("[Edition] Pro モードで実行します (AdvancedExtractor 有効)")
        except Exception as adv_err:  # pragma: no cover
            logging.error(
                f"Pro モードを要求されましたが AdvancedExtractor を初期化できませんでした: {adv_err}"
            )
            logging.error("free モードにフォールバックします。")
            args.edition = "free"
    
    if args.edition == "free":
        simple_extractor = SimpleExtractor(patent_mode=args.patent)
        logging.info("[Edition] Free モードで実行します (SimpleExtractor 有効)")


    total_para = 0

    for pdf_path_str in args.pdf:
        pdf_path = Path(pdf_path_str)
        if not pdf_path.exists():
            logging.error(f"ファイルが見つかりません: {pdf_path}")
            continue
        
        logging.info(f"処理開始: {pdf_path.name}")

        paras = []
        try:
            if args.edition == "pro" and advanced_extractor:
                paras = advanced_extractor.process_pdf(pdf_path, force_ocr=args.force_ocr)
            elif args.edition == "free" and simple_extractor:
                 paras = simple_extractor.process_pdf(pdf_path)
        except Exception as e:
            logging.error(f"{pdf_path.name} の抽出中にエラーが発生しました: {e}", exc_info=True)
            continue

        try:
            doc_id = pdf_path.stem
            save_paragraphs_to_db(args.db_path, doc_id, paras)
            logging.info(f"{pdf_path.name}: {len(paras)} 段落を保存しました。")
            total_para += len(paras)
        except Exception as e:
            logging.error(f"{pdf_path.name} のDB保存中にエラーが発生しました: {e}", exc_info=True)

    logging.info(f"処理完了。合計 {total_para} 段落を保存しました。")


if __name__ == "__main__":
    main() 