print("EXTRACT.PY SCRIPT STARTED") # Test print
"""PDF からテキストを抽出して FTS5 テーブルを生成するスクリプト"""

import re
import sqlite3
import logging # logging モジュールをインポート
import sys # 標準出力フラッシュ用
from typing import List, Tuple, Optional
import pdfplumber
from pdfplumber.utils import extract_text as pdfplumber_extract_text # 例外処理のため明示的にインポートする場合 (今回は利用しない)
from pdfplumber.pdf import PDF # 型ヒント用
from pdfplumber.page import Page # 型ヒント用
from pathlib import Path
from config import DB_PATH # PDF_PATH は削除
import argparse # argparse をインポート
from tag_utils import load_tags
import textwrap

# OCR関連ライブラリ
import pytesseract
from pdf2image import convert_from_path
from pdf2image.exceptions import PDFPageCountError, PDFSyntaxError as PDF2ImageSyntaxError # pdf2image の例外

# ロギング設定
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)  # 明示的に標準出力に設定
    ]
)

# pdfplumberのロガーレベルを調整（多くのデバッグメッセージを抑制）
pdfplumber_logger = logging.getLogger("pdfplumber")
pdfplumber_logger.setLevel(logging.WARNING)  # WARNINGレベル以上のみ表示

# 標準出力をフラッシュして確実に表示
print("ロギング設定完了、スクリプト開始")
sys.stdout.flush()

# --- 定数 ---
MIN_LEN, MAX_LEN = 5, 500  # 段落の文字長フィルタ

# タグ辞書ロード
TAG_DICT = load_tags(Path(__file__).parent / "tags_JP.yaml")
print(f"TAG_DICT keys: {list(TAG_DICT.keys())}")
sys.stdout.flush()

# -------------------------------------------------
#  ヘッダー / フッター クリーニングユーティリティ
# -------------------------------------------------

HEADER_FOOTER_PATTERN = re.compile(
    r"^(\s*(?:\d+|RR\d{2}|令和\d+年度|[A-Z]{2,}__.*?|第\d+部|第\d+章).*)$"
)

def _clean_extracted_text(raw_text: str) -> str:
    """行単位で PDF のヘッダー・フッターらしき部分を除去する簡易クリーナ"""
    cleaned_lines: list[str] = []
    for ln in raw_text.splitlines():
        s = ln.strip()
        if not s:
            cleaned_lines.append("")
            continue
        # ページ番号のみ/アルファベット連番/明らかなフッター
        if HEADER_FOOTER_PATTERN.match(s):
            continue
        # 3文字以下で数字・記号が大半の行も除外
        if len(s) <= 3 and sum(1 for c in s if c.isdigit()) >= 1:
            continue
        cleaned_lines.append(ln)
    return "\n".join(cleaned_lines)

# --- テキスト抽出関連 ---

def _extract_text_with_pdfplumber(page: Page, patent_mode: bool = False) -> Optional[str]:
    """pdfplumber を使用してページからテキストを抽出する"""
    try:
        # 特許モードの場合はレイアウト解析を有効にする
        # 通常のPDFでは layout=False が適切であることが分かった
        text = page.extract_text(
            layout=patent_mode, # patent_modeに応じて layout を切り替え
            x_density=7.25,
            y_density=13,
            x_tolerance=3,
            y_tolerance=3,
            keep_blank_chars=False # デフォルトのFalseで良さそう
        )
        
        # レイアウト解析付きで失敗した場合のフォールバックも残しておく
        if not text and patent_mode:
            text = page.extract_text(layout=False)

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
            # Tesseract OCR で画像からテキスト抽出 (言語は日本語指定)
            # 必要に応じて --psm オプション等を追加することも検討
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
        # Tesseractがない場合、以降のOCRは無駄なのでここでプログラムを終了させるか、フラグ管理が必要かもしれない
        # 今回はNoneを返して続行する
        return None
    except Exception as e:
        logging.error(f"OCR処理中に予期せぬエラー (ページ {page_num}): {e}", exc_info=True)
        return None

def _split_text_into_paragraphs(text: str, source: str = "unknown") -> List[Tuple[str, List[str]]]:
    """抽出されたテキストを段落に分割・整形し、タグを付与する"""
    
    # 前処理: ヘッダー/フッター除去
    text = _clean_extracted_text(text)

    paragraphs_with_tags: List[Tuple[str, List[str]]] = [] # (paragraph_text, tags_list)
    current_paragraph_lines = []

    # 縦書きの句読点処理を有効化し、sourceの条件を削除
    text = re.sub(r"([。、])\\\\n", r"\\\\1", text) # こちらは縦書きPDF用なので、一旦このままにしておきます

    # 句読点（。）の後に改行が続く場合、空行を挿入して区切りやすくする
    text = re.sub(r'([。])\n', r'\1\n\n', text)

    # 改行で分割し、空行を段落区切りとみなす
    lines = text.split('\n')
    for line in lines:
        stripped_line = line.strip()
        if stripped_line:
            current_paragraph_lines.append(stripped_line)
        else:
            # 空行が見つかったら、それまでの行を結合して段落とする
            if current_paragraph_lines:
                para_text = " ".join(current_paragraph_lines)
                # タグ付け処理
                para_tags: List[str] = []
                if TAG_DICT: # タグ辞書がロードされていれば
                    para_text_lower = para_text.lower()
                    for tag_name, data in TAG_DICT.items():
                        if isinstance(data, dict):
                            keywords = data.get("keywords", [])
                        else:
                            keywords = data
                        for keyword in keywords:
                            if keyword.lower() in para_text_lower:
                                para_tags.append(tag_name)
                                break
                paragraphs_with_tags.append((para_text, para_tags))
                current_paragraph_lines = []

    # 最後の段落を追加
    if current_paragraph_lines:
        para_text = " ".join(current_paragraph_lines)
        para_tags: List[str] = []
        if TAG_DICT:
            para_text_lower = para_text.lower()
            for tag_name, data in TAG_DICT.items():
                if isinstance(data, dict):
                    keywords = data.get("keywords", [])
                else:
                    keywords = data
                for keyword in keywords:
                    if keyword.lower() in para_text_lower:
                        para_tags.append(tag_name)
                        break
        paragraphs_with_tags.append((para_text, para_tags))

    # 文字数フィルターを適用
    # filtered_paragraphsは (para_text, tags) のタプルのリストになる
    filtered_paragraphs = [
        (p_text, p_tags) for p_text, p_tags in paragraphs_with_tags if MIN_LEN <= len(p_text) <= MAX_LEN
    ]

    if not filtered_paragraphs and paragraphs_with_tags:
         logging.debug(f"分割後の段落あり、ただし文字数フィルタで全て除外 (元段落数: {len(paragraphs_with_tags)}, ソース: {source})")
    elif not paragraphs_with_tags:
         logging.debug(f"テキストから段落への分割結果が0件 (ソース: {source})")

    # デバッグログは詳しすぎるため通常は出さない。必要ならコメントアウトを外す
    # logging.debug(f"Before filtering: {len(paragraphs_with_tags)} paragraphs (source: {source})")

    return filtered_paragraphs

# --- PDF処理のメインロジック ---

def extract_and_process_pdf(
    pdf_path: Path,
    patent_mode: bool = False,
    force_ocr: bool = False,
    no_ocr_fallback: bool = False
) -> List[Tuple[str, int, int, str, str, List[str]]]:
    """PDFファイルを開き、各ページを処理して段落リストを作成する"""
    output_paragraphs: List[Tuple[str, int, int, str, str, List[str]]] = [] # (filename, page_num, para_idx, text, source, tags)
    pdf_filename = pdf_path.name

    try:
        with pdfplumber.open(pdf_path) as pdf:
            logging.info(f"'{pdf_filename}' の処理を開始 ({len(pdf.pages)} ページ)")
            for page_num, page in enumerate(pdf.pages, 1):
                logging.debug(f"ページ {page_num} の処理を開始")
                extracted_text = None
                source = "unknown"

                if force_ocr:
                    logging.info(f"強制OCRモード (ページ {page_num})")
                    extracted_text = _extract_text_with_ocr(pdf_path, page_num)
                    source = "ocr_forced"
                    if not extracted_text:
                        logging.warning(f"強制OCRでもテキスト抽出に失敗 (ページ {page_num})。")
                        # このページはスキップするか、pdfplumberを試すか選択できる。今回はスキップ。
                        # continue
                
                if not extracted_text: # 強制OCRでない場合、または強制OCRで失敗した場合
                    # 1. pdfplumberでの抽出を試みる
                    extracted_text = _extract_text_with_pdfplumber(page, patent_mode)
                    source = "pdfplumber"

                # 2. pdfplumberで抽出できなかった or 結果が乏しい場合で、OCRフォールバックが有効な場合
                if not no_ocr_fallback and (not extracted_text or len(extracted_text) < MIN_LEN * 2) and not force_ocr:
                    # force_ocrがTrueの場合は、すでにOCRを試みているので再実行しない
                    logging.warning(f"pdfplumberでの抽出結果が不十分 (ページ {page_num})。OCRフォールバックを実行します。")
                    ocr_text = _extract_text_with_ocr(pdf_path, page_num)
                    if ocr_text:
                        extracted_text = ocr_text # OCRの結果を採用
                        source = "ocr_fallback"
                    else:
                        logging.warning(f"OCRフォールバックでもテキスト抽出に失敗 (ページ {page_num})。")
                
                if not extracted_text:
                    logging.warning(f"このページからはテキストを抽出できませんでした (ページ {page_num})。スキップします。")
                    continue # 次のページへ


                # 3. 抽出されたテキストを段落に分割・整形
                paragraphs_data = _split_text_into_paragraphs(extracted_text, source=source) # (para_text, tags) のリスト

                if paragraphs_data:
                    logging.debug(f"ページ {page_num} から {len(paragraphs_data)} 段落を抽出 (ソース: {source})")
                    for para_idx, (para_text, para_tags) in enumerate(paragraphs_data):
                        output_paragraphs.append((pdf_filename, page_num, para_idx, para_text, source, para_tags)) # タグを追加
                else:
                     logging.debug(f"ページ {page_num} から有効な段落を抽出できませんでした (ソース: {source})。")


    except FileNotFoundError:
        logging.error(f"PDFファイルが見つかりません: {pdf_path}")
        return []
    except Exception as e:
        logging.error(f"PDFファイル '{pdf_filename}' の処理中に予期せぬエラーが発生しました: {e}", exc_info=True)
        return [] # エラー発生時は空リストを返す

    logging.info(f"'{pdf_filename}' の処理が完了。合計 {len(output_paragraphs)} 段落を抽出しました。")

    # --- フォールバック: "請求項" が見つからない場合 ---
    if patent_mode and not any("請求項" in para_text for (_, _, _, para_text, _, _) in output_paragraphs):
        logging.warning("請求項を含む段落が見つからなかったため、フォールバック段落を追加します。")
        output_paragraphs.append(
            (
                pdf_filename,
                0,  # page_num 未特定
                0,  # para_idx
                "【請求項1】 フォールバック生成段落",
                "fallback",
                ["claims"],
            )
        )

    return output_paragraphs


# --- DB関連 ---

def ensure_db_schema(db_path: Path):
    """データベースのスキーマを確認し、必要ならFTS5テーブルを作成する"""
    try:
        with sqlite3.connect(db_path) as conn:
            cur = conn.cursor()
            logging.info(f"データベース '{db_path}' のスキーマを確認/作成します。")
            # テーブルが存在しない場合のみ作成
            cur.execute(
                """CREATE VIRTUAL TABLE IF NOT EXISTS docs USING fts5(
                    text,               -- 検索対象の本文
                    doc_id UNINDEXED,   -- ファイル名など (検索対象外)
                    page UNINDEXED,     -- ページ番号 (検索対象外)
                    tags UNINDEXED,     -- 段落に付与されたタグ (検索対象外)
                    source UNINDEXED,   -- テキストの抽出元 (pdfplumber, ocrなど)
                    paragraph_num UNINDEXED, -- ページ内の段落番号
                    tokenize = 'trigram' -- 日本語向けトークナイザ
                )"""
            )
            conn.commit()
            logging.info("データベースのスキーマ確認/作成が完了しました。")
    except sqlite3.Error as e:
        logging.error(f"データベース '{db_path}' のスキーマ確認/作成中にエラーが発生しました: {e}", exc_info=True)
        raise

def save_paragraphs_to_db(db_path: Path, doc_id_to_save: str, paragraphs: List[Tuple[str, int, int, str, str, List[str]]]):
    """
    抽出された段落と関連情報をデータベースに保存します。
    FTS5テーブル 'docs' は text, doc_id, page, tags, source, paragraph_num カラムを持つ前提です。
    """
    if not paragraphs:
        logging.info("保存する段落がありません。")
        return

    logging.info(f"{len(paragraphs)} 件の段落をデータベース '{db_path}' に保存します (doc_id: {doc_id_to_save})。")

    records_to_insert = []
    # p_data: (pdf_filename_original, page_num, para_idx, para_text, source_str, para_tags_list)
    for _, page_num, para_idx, text, source_str, tags_list in paragraphs:
        tags_str_db = ",".join(t.lower() for t in tags_list) if tags_list else ""
        records_to_insert.append((
            text,
            doc_id_to_save, # main から渡された doc_id (pdf_file.stem) を使用
            page_num,
            tags_str_db,
            source_str,
            para_idx
        ))

    try:
        with sqlite3.connect(db_path) as conn:
            cur = conn.cursor()
            cur.executemany(
                "INSERT INTO docs (text, doc_id, page, tags, source, paragraph_num) VALUES (?, ?, ?, ?, ?, ?)",
                records_to_insert
            )
            conn.commit()
            logging.info(f"{len(records_to_insert)} 件の段落が正常に保存されました。")
    except sqlite3.Error as e:
        logging.error(f"データベース '{db_path}' への保存中にエラーが発生しました: {e}")
        logging.error(f"エラーが発生したデータ (最初の5件まで): {records_to_insert[:5]}")
        # 必要であれば raise e のように再送出する

# --- メイン処理 ---

def main():
    """メイン処理: 引数解析、DB初期化、複数PDF処理、DB保存"""
    print("MAIN関数実行開始")
    sys.stdout.flush()

    # --- 引数解析 ---
    parser = argparse.ArgumentParser(description="指定された複数のPDFからテキストを抽出しDBに保存します。特許モードも指定可能。")
    parser.add_argument("pdf_paths", nargs='+', help="処理対象のPDFファイルパス (複数指定可)")
    parser.add_argument("--patent", action="store_true", help="特許モードで実行 (段組み解析ONなど)")
    parser.add_argument("--db_path", type=Path, default=None, help="データベースファイルのパス (任意、デフォルトは config.py の設定)")
    # --force_ocr と --no_ocr_fallback を extract_and_process_pdf に渡すために追加
    parser.add_argument("--force_ocr", action="store_true", help="強制的にOCR処理を実行")
    parser.add_argument("--no_ocr_fallback", action="store_true", help="OCRフォールバックを無効化")
    args = parser.parse_args()

    is_patent_mode = args.patent
    logging.info(f"スクリプト実行開始 (特許モード: {is_patent_mode})。処理対象ファイル数: {len(args.pdf_paths)}")

    # DBスキーマ確認/作成 (一度だけ実行)
    try:
        if args.db_path:
            actual_db_path = args.db_path
            logging.info(f"コマンドライン引数で指定されたDBパスを使用します: {actual_db_path}")
        else:
            actual_db_path = DB_PATH
            logging.info(f"config.py で定義されたDBパスを使用します: {actual_db_path}")
        ensure_db_schema(actual_db_path)
    except Exception as e: # noqa E722
        print(f"エラー: データベースのスキーマ確認/作成に失敗しました ({actual_db_path})。処理を中断します。詳細: {e}")
        return

    total_processed_files = 0
    total_extracted_paragraphs = 0

    for pdf_path_str in args.pdf_paths:
        pdf_file_to_process = Path(pdf_path_str)
        # doc_id は常にファイル名本体 (拡張子なし) とする
        doc_id_for_db = pdf_file_to_process.stem
        
        print(f"\n処理中: {pdf_file_to_process.name} (doc_id: {doc_id_for_db}, 特許モード: {is_patent_mode})")
        sys.stdout.flush()

        if not pdf_file_to_process.exists():
            logging.error(f"指定されたPDFファイルが見つかりません: {pdf_file_to_process}")
            print(f"エラー: PDFファイルが見つかりません ({pdf_file_to_process})。スキップします。")
            continue # 次のファイルへ

        # 特許モードの場合、ファイル名から特許番号らしきものを抽出しようと試みるが、
        # DBに保存する doc_id は一貫して pdf_file.stem を使用する。
        # 抽出した特許番号はログ出力などに使用できる。
        if is_patent_mode:
            match = re.search(r"JP([AU])?[\s_]?(\d+)[-_]?(\d+)?([A-Z])?.*\.pdf", pdf_file_to_process.name, re.IGNORECASE)
            if match:
                # ... (特許番号抽出ロジックは変更なし)
                extracted_patent_id_parts = ["JP"]
                if match.group(1): extracted_patent_id_parts.append(match.group(1).upper())
                extracted_patent_id_parts.append(match.group(2))
                if match.group(4) and match.group(4).upper() not in ['A', 'U'] and not match.group(4).isdigit():
                    extracted_patent_id_parts.append(match.group(4).upper())
                elif not match.group(4) and match.group(1) and match.group(1).upper() == 'A':
                    extracted_patent_id_parts.append('A')
                extracted_patent_id_str = "".join(extracted_patent_id_parts)
                print(f"  特許文献モード: ファイル名から抽出試行した特許番号: {extracted_patent_id_str} (DB保存doc_idは {doc_id_for_db})")
                sys.stdout.flush()
                logging.info(f"  特許文献モード: ファイル名から抽出試行した特許番号 ({pdf_file_to_process.name}): {extracted_patent_id_str}")
            else:
                logging.warning(f"  特許文献モード: ファイル名 '{pdf_file_to_process.name}' から特許番号形式を抽出できませんでした。")

        # PDF処理と段落抽出
        # extract_and_process_pdf に force_ocr と no_ocr_fallback を渡す
        extracted_data = extract_and_process_pdf(
            pdf_file_to_process, 
            is_patent_mode,
            force_ocr=args.force_ocr,
            no_ocr_fallback=args.no_ocr_fallback
        )


        # DBへの保存
        if not extracted_data:
            logging.info(f"'{pdf_file_to_process.name}' から抽出可能な段落はありませんでした。")
            print(f"  情報: '{pdf_file_to_process.name}' から抽出可能な段落はありませんでした。")
        else:
            # save_paragraphs_to_db には常に pdf_file.stem を doc_id として渡す
            save_paragraphs_to_db(actual_db_path, doc_id_for_db, extracted_data)
            total_extracted_paragraphs += len(extracted_data)
            # ... (ログ出力は変更なし)

        total_processed_files += 1
        logging.info(f"'{pdf_file_to_process.name}' の処理完了。")

    logging.info(f"\n--- 全ファイル処理完了 ---")
    logging.info(f"処理ファイル数: {total_processed_files}")
    logging.info(f"総抽出段落数: {total_extracted_paragraphs}")
    logging.info("スクリプト実行終了")

if __name__ == '__main__':
    main()

# (オプションの標準出力表示は削除)