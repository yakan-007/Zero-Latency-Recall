print("EXTRACT_NEW.PY SCRIPT STARTED") # Test print

"""PDF からテキストを抽出して FTS5 テーブルを生成するスクリプト"""

import re
import sqlite3
import logging # logging モジュールをインポート
from typing import List, Tuple, Optional
import pdfplumber
from pdfplumber.page import Page # 型ヒント用
from pathlib import Path
from config import DB_PATH # PDF_PATH は削除
import argparse # argparse をインポート

# OCR関連ライブラリ
import pytesseract
from pdf2image import convert_from_path
from pdf2image.exceptions import PDFPageCountError, PDFSyntaxError as PDF2ImageSyntaxError # pdf2image の例外

# ロギング設定
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# --- 定数 ---
MIN_LEN, MAX_LEN = 10, 200  # 段落の文字長フィルタ

# --- テキスト抽出関連 ---

def _extract_text_with_pdfplumber(page: Page, patent_mode: bool = False) -> Optional[str]:
    """pdfplumber を使用してページからテキストを抽出する"""
    try:
        # 特許モードの場合はレイアウト解析を有効にする
        # x_density, y_density は pdfplumber のデフォルトに近い値や、試行して調整した値
        text = page.extract_text(
            layout=patent_mode, # patent_modeに応じて layout を切り替え
            x_density=7.25,
            y_density=13,
            x_tolerance=3, # 文字間の水平方向の許容範囲 (デフォルトは3)
            y_tolerance=3  # 文字間の垂直方向の許容範囲 (デフォルトは3)
        )
        if text:
            text = text.strip()
            logging.debug(f"pdfplumber で抽出成功 (layout={patent_mode}): {text[:100]}...")
            return text
        else:
            logging.debug(f"pdfplumber でテキスト抽出できず (layout={patent_mode})")
            return None
    except Exception as e:
        logging.warning(f"pdfplumber でのテキスト抽出中にエラー: {e}")
        return None

def _extract_text_with_ocr(pdf_path: Path, page_num: int, lang: str = 'jpn') -> Optional[str]:
    """OCR (Tesseract) を使用して指定されたPDFページからテキストを抽出する"""
    logging.info(f"OCR処理を開始 (ページ {page_num}) ...")
    try:
        images = convert_from_path(pdf_path, first_page=page_num, last_page=page_num, fmt='png', thread_count=1)
        if images:
            text = pytesseract.image_to_string(images[0], lang=lang)
            logging.info(f"OCR処理が完了 (ページ {page_num})")
            return text.strip() if text else None
        else:
            logging.warning(f"pdf2imageで画像への変換に失敗 (ページ {page_num})")
            return None
    except (PDFPageCountError, PDF2ImageSyntaxError) as e:
        logging.error(f"pdf2imageでの画像変換中にエラー (ページ {page_num}): {e}")
        return None
    except pytesseract.TesseractNotFoundError:
        logging.error("Tesseract OCRが見つかりません。インストールされているか、PATHを確認してください。", exc_info=True)
        return None
    except Exception as e:
        logging.error(f"OCR処理中に予期せぬエラー (ページ {page_num}): {e}", exc_info=True)
        return None

def _split_text_into_paragraphs(text: str, source: str = "unknown") -> List[str]:
    """抽出されたテキストを段落に分割・整形する"""
    paragraphs = []
    current_paragraph_lines = []

    # 縦書きの句読点処理を有効化
    text = re.sub(r"([。、])\n", r"\1", text)

    # 改行で分割し、空行を段落区切りとみなす
    lines = text.split('\n')
    for line in lines:
        stripped_line = line.strip()
        if stripped_line:
            current_paragraph_lines.append(stripped_line)
        else:
            # 空行が見つかったら、それまでの行を結合して段落とする
            if current_paragraph_lines:
                paragraphs.append(" ".join(current_paragraph_lines))
                current_paragraph_lines = []

    # 最後の段落を追加
    if current_paragraph_lines:
        paragraphs.append(" ".join(current_paragraph_lines))

    # 文字数フィルターを適用
    filtered_paragraphs = [
        p for p in paragraphs if MIN_LEN <= len(p) <= MAX_LEN
    ]

    return filtered_paragraphs

# --- PDF処理のメインロジック ---

def extract_and_process_pdf(pdf_path: Path, patent_mode: bool = False) -> List[Tuple[str, int, int, str, str]]:
    """PDFファイルを開き、各ページを処理して段落リストを作成する"""
    output_paragraphs: List[Tuple[str, int, int, str, str]] = []
    pdf_filename = pdf_path.name

    try:
        with pdfplumber.open(pdf_path) as pdf:
            logging.info(f"'{pdf_filename}' の処理を開始 ({len(pdf.pages)} ページ)")
            for page_num, page in enumerate(pdf.pages, 1):
                logging.debug(f"ページ {page_num} の処理を開始")

                # 1. pdfplumberでの抽出を試みる
                extracted_text = _extract_text_with_pdfplumber(page, patent_mode)
                source = "pdfplumber"

                # 2. pdfplumberで抽出できなかった or 結果が乏しい場合、OCRを試みる
                if not extracted_text or len(extracted_text) < MIN_LEN * 2:
                    logging.warning(f"pdfplumberでの抽出結果が不十分 (ページ {page_num})。OCRフォールバックを実行します。")
                    extracted_text = _extract_text_with_ocr(pdf_path, page_num)
                    source = "ocr"
                    if not extracted_text:
                        logging.warning(f"OCRでもテキスト抽出に失敗 (ページ {page_num})。このページはスキップします。")
                        continue

                # 3. 抽出されたテキストを段落に分割・整形
                paragraphs = _split_text_into_paragraphs(extracted_text, source=source)

                if paragraphs:
                    logging.debug(f"ページ {page_num} から {len(paragraphs)} 段落を抽出 (ソース: {source})")
                    for para_idx, para_text in enumerate(paragraphs):
                        output_paragraphs.append((pdf_filename, page_num, para_idx, para_text, source))
                else:
                     logging.debug(f"ページ {page_num} から有効な段落を抽出できませんでした (ソース: {source})。")

    except FileNotFoundError:
        logging.error(f"PDFファイルが見つかりません: {pdf_path}")
        return []
    except Exception as e:
        logging.error(f"PDFファイル '{pdf_filename}' の処理中に予期せぬエラーが発生しました: {e}", exc_info=True)
        return []

    logging.info(f"'{pdf_filename}' の処理が完了。合計 {len(output_paragraphs)} 段落を抽出しました。")
    return output_paragraphs

# --- DB関連 ---

def init_db(db_path: Path):
    """データベースを初期化し、FTS5テーブルを作成する"""
    try:
        with sqlite3.connect(db_path) as conn:
            cur = conn.cursor()
            logging.info(f"データベース '{db_path}' のテーブルを初期化します。")
            cur.execute("DROP TABLE IF EXISTS paragraphs")
            # FTS5 テーブル (trigram トークナイザ)
            cur.execute(
                """CREATE VIRTUAL TABLE paragraphs USING fts5(
                    pdf_filename,
                    page,
                    paragraph_num,
                    content,
                    source,
                    tokenize = 'trigram'
                )"""
            )
            conn.commit()
            logging.info("データベースの初期化が完了しました。")
    except sqlite3.Error as e:
        logging.error(f"データベース '{db_path}' の初期化中にエラーが発生しました: {e}", exc_info=True)
        raise

def save_paragraphs_to_db(db_path: Path, paragraphs: List[Tuple[str, int, int, str, str]]):
    """抽出された段落をデータベースに保存する"""
    if not paragraphs:
        logging.info("保存する段落がありませんでした。")
        return

    logging.info(f"{len(paragraphs)} 件の段落をデータベース '{db_path}' に保存します。")
    try:
        with sqlite3.connect(db_path) as conn:
            conn.executemany(
                "INSERT INTO paragraphs (pdf_filename, page, paragraph_num, content, source) VALUES (?, ?, ?, ?, ?)", 
                paragraphs
            )
            conn.commit()
        logging.info(f"{len(paragraphs)} 件の段落が正常に保存されました。")
    except sqlite3.Error as e:
        logging.error(f"データベース '{db_path}' への保存中にエラーが発生しました: {e}")

# --- メイン処理 ---

def main():
    """メイン処理: 引数解析、DB初期化、PDF処理、DB保存"""
    print("MAIN FUNCTION STARTED") # 確認用print
    
    # --- 引数解析 ---
    parser = argparse.ArgumentParser(description="PDFからテキストを抽出しDBに保存します。特許モードも指定可能。")
    parser.add_argument("pdf_path", type=str, help="処理対象のPDFファイルパス")
    parser.add_argument("--patent", action="store_true", help="特許モードで実行 (段組み解析ONなど)")
    args = parser.parse_args()

    # Pathオブジェクトに変換
    pdf_file_to_process = Path(args.pdf_path)
    is_patent_mode = args.patent

    print(f"PDF path: {pdf_file_to_process}, Patent mode: {is_patent_mode}") # 確認用print
    logging.info(f"スクリプト実行開始 (特許モード: {is_patent_mode})")

    # doc_id の抽出とログ出力 (特許モード時のみ)
    doc_id = None
    if is_patent_mode:
        # ファイル名から公報番号らしきものを抽出
        match = re.search(r"JP([AU])?[\s_]?(\d+)[-_]?(\d+)?([A-Z])?.*\.pdf", pdf_file_to_process.name, re.IGNORECASE)
        if match:
            doc_id_parts = ["JP"]
            if match.group(1): # A or U
                doc_id_parts.append(match.group(1).upper())
            doc_id_parts.append(match.group(2)) # 主番号
            if match.group(4) and match.group(4).upper() not in ['A', 'U'] and not match.group(4).isdigit():
                 doc_id_parts.append(match.group(4).upper())
            elif not match.group(4) and match.group(1) and match.group(1).upper() == 'A':
                doc_id_parts.append('A')
            doc_id = "".join(doc_id_parts)
            print(f"Extracted doc_id: {doc_id}") # 確認用print
            logging.info(f"抽出された doc_id: {doc_id}")
        else:
            logging.warning(f"ファイル名 '{pdf_file_to_process.name}' からdoc_idを抽出できませんでした。")

    # PDFパスの存在確認
    if not pdf_file_to_process.exists():
        logging.error(f"指定されたPDFファイルが見つかりません: {pdf_file_to_process}")
        print(f"エラー: 指定されたPDFファイルが見つかりません ({pdf_file_to_process})。処理を中断します。")
        return

    # DB初期化
    try:
        init_db(DB_PATH)
    except Exception:
        print(f"エラー: データベースの初期化に失敗しました ({DB_PATH})。処理を中断します。")
        return

    # PDF処理と段落抽出
    extracted_data = extract_and_process_pdf(pdf_file_to_process, is_patent_mode)

    # DBへの保存
    if not extracted_data:
        logging.info(f"'{pdf_file_to_process.name}' から抽出可能な段落はありませんでした。")
        print(f"情報: '{pdf_file_to_process.name}' から抽出可能な段落はありませんでした。")
    else:
        logging.debug(f"First item in all_paragraphs before saving: {extracted_data[0]}")
        logging.debug(f"Length of first item tuple: {len(extracted_data[0])}")
        save_paragraphs_to_db(DB_PATH, extracted_data)

    # サンプル表示
    logging.info("\n--- Extracted Paragraphs (Sample) ---")
    for i, (pdf_filename, page_num, para_idx, content, source) in enumerate(extracted_data[:5]):
        logging.info(f"[{pdf_filename}:{page_num}-{para_idx} ({source})] {content}")
    if len(extracted_data) > 5:
        logging.info(f"... (他 {len(extracted_data) - 5} 件)")
    logging.info("--- End of Paragraphs Sample ---")
    logging.info("スクリプト実行終了")

if __name__ == '__main__':
    main() 