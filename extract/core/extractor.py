"""PDF からテキストを抽出して FTS5 テーブルを生成するスクリプト"""

import re
import logging # logging モジュールをインポート
from typing import List, Tuple, Optional, Dict
import pdfplumber
from pdfplumber.utils import extract_text as pdfplumber_extract_text
from pdfplumber.page import Page
from pathlib import Path
import sys # Keep sys for logging handler
import textwrap
import abc  # 追加: Strategy 用抽象基底クラス
from .strategies import BaseExtractor, SimpleExtractor, AdvancedExtractor

# PyMuPDF (fitz) のインポート試行とフラグ設定
_FITZ_AVAILABLE = False # Default to False
try:
    import fitz  # type: ignore
    logging.debug("PyMuPDF (fitz) imported successfully for general use globally.")
    _FITZ_AVAILABLE = True
except ImportError:
    logging.warning("PyMuPDF (fitz) not found globally. Fitz-dependent functions will not be available.")
except Exception as e:
    logging.error(f"An unexpected error occurred during fitz import: {e}")

logging.debug(f"_FITZ_AVAILABLE set to: {_FITZ_AVAILABLE} at global scope")

# Correct imports for config and utils
try:
    from ..config import DB_PATH  # Relative import from parent directory
except (ModuleNotFoundError, ImportError): # Wider catch for potential import issues
    DB_PATH = Path("zlr.sqlite")
    logging.warning("../config.py が見つからないため、DB_PATH をデフォルト 'zlr.sqlite' に設定します。")

# Remove argparse import
from .utils import load_tags # Relative import from current directory


# OCR関連ライブラリ (Keep top-level for now, consider lazy import later)
import pytesseract
from pdf2image import convert_from_path
from pdf2image.exceptions import PDFPageCountError, PDFSyntaxError as PDF2ImageSyntaxError, PDFInfoNotInstalledError

# --- 定数 ---
MIN_LEN, MAX_LEN = 5, 6000

# タグ辞書ロード
# Use absolute path based on this file's location to find tags_JP.yaml
_CURRENT_DIR = Path(__file__).parent
TAGS_YAML_PATH = _CURRENT_DIR.parent / "tags_JP.yaml" # Assumes tags_JP.yaml is in the parent 'extract' directory
TAG_DICT = {}
if TAGS_YAML_PATH.exists():
    TAG_DICT = load_tags(TAGS_YAML_PATH)
    logging.info(f"タグ辞書をロードしました: {TAGS_YAML_PATH}")
    # logging.debug(f"TAG_DICT keys: {list(TAG_DICT.keys())}") # Debug level
else:
    logging.warning(f"タグ辞書ファイルが見つかりません: {TAGS_YAML_PATH}")

# Remove print/flush for TAG_DICT keys

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

# -------------------------------------------------
#  テキスト正規化ユーティリティ（比較用）
# -------------------------------------------------

# 連続スペースや改行を潰してテキスト比較を安定化させる。

_re_multi_space = re.compile(r"[ \t]{2,}")
_re_multi_newline = re.compile(r"\n{2,}")


def _normalize_text_for_compare(text: str) -> str:
    """不要な連続スペース・改行を圧縮して比較しやすく整形"""
    text = _re_multi_space.sub(" ", text)
    text = _re_multi_newline.sub("\n", text)
    return text.strip()

# --- テキスト抽出関連 ---

def _is_likely_multicolumn(page: Page, central_band_ratio: float = 0.15, min_words_for_check: int = 20) -> bool:
    """
    Heuristic to determine if a page is likely multi-column based on word distribution.
    Checks for a relatively empty vertical band in the center of the page.
    """
    pdf_filename = Path(page.pdf.stream.name).name # Get filename for logging
    try:
        # Use slightly more tolerant settings for word extraction for this check
        words = page.extract_words(x_tolerance=3, y_tolerance=3, keep_blank_chars=False, use_text_flow=True)
    except Exception as e:
        logging.debug(f"[{pdf_filename} - Page {page.page_number}] _is_likely_multicolumn: extract_words failed: {e}")
        return False

    if not words or len(words) < min_words_for_check:
        logging.debug(f"[{pdf_filename} - Page {page.page_number}] _is_likely_multicolumn: Too few words ({len(words)}) for check, assuming not multicolumn.")
        return False

    page_width = page.width
    center_x = page_width / 2
    band_half_width = (page_width * central_band_ratio) / 2
    central_band_start = center_x - band_half_width
    central_band_end = center_x + band_half_width

    words_in_center_band = 0
    words_left_of_band = 0
    words_right_of_band = 0
    
    for word in words:
        word_center_x = (word['x0'] + word['x1']) / 2
        if word_center_x < central_band_start:
            words_left_of_band += 1
        elif word_center_x >= central_band_end:
            words_right_of_band += 1
        else:
            words_in_center_band += 1
            
    if words_left_of_band == 0 and words_right_of_band == 0:
        is_multicolumn = False
    else:
        min_side_words = min_words_for_check * 0.1 
        center_sparsity_threshold = 0.65 # 0.55 から 0.65 へ変更
        sides_are_populated = words_left_of_band > min_side_words and words_right_of_band > min_side_words

        if sides_are_populated:
            avg_side_words = (words_left_of_band + words_right_of_band) / 2
            if avg_side_words > 0:
                 is_multicolumn = (words_in_center_band / avg_side_words) < center_sparsity_threshold
            else:
                 is_multicolumn = False
        else:
            is_multicolumn = False

    logging.debug(
        f"[{pdf_filename} - Page {page.page_number}] _is_likely_multicolumn: W={page_width:.0f}, "
        f"Band ({central_band_start:.0f}-{central_band_end:.0f}). "
        f"Words L={words_left_of_band}, C={words_in_center_band}, R={words_right_of_band}. "
        f"Result -> {is_multicolumn}"
    )
    return is_multicolumn

def _extract_text_with_pdfplumber(page: Page, patent_mode: bool = False) -> Tuple[Optional[str], str]:
    """pdfplumber を使用してページからテキストを抽出する

    改良点:
    ・patent_mode=False の通常モードでも layout=True も試し、抽出文字数が多い方を採用することで
      2カラム文書の読み取り精度向上を図る。
    
    返り値:
        Tuple[Optional[str], str]: (抽出されたテキスト, 採用されたメソッド情報)
    """
    pdf_filename = Path(page.pdf.stream.name).name # Get filename for logging
    chosen_method_info = "unknown_initial" 
    text_to_return: Optional[str] = None

    try:
        if patent_mode:
            text = page.extract_text(
                layout=True,
                x_density=7.25,
                y_density=13,
                x_tolerance=3,
                y_tolerance=3,
                keep_blank_chars=False,
            )
            if not text:
                text = page.extract_text(layout=False)
            chosen_method_info = "patent_layout_true_fallback_false" if text else "patent_layout_false"
            text_to_return = text
        else:
            is_multi_fast = _is_likely_multicolumn(page, central_band_ratio=0.25)
            if is_multi_fast:
                # Pass the document object for fitz operations
                fitz_doc = None
                if _FITZ_AVAILABLE:
                    try:
                        # This still opens the doc per page, needs refactoring in the caller
                        fitz_doc = fitz.open(page.pdf.stream.name)
                    except Exception as e:
                        logging.warning(f"[{pdf_filename} - Page {page.page_number}] Failed to open with fitz for fast check: {e}")
                
                fast_fitz_text = _safe_multicol_fitz(page, fitz_doc) # Pass fitz_doc
                if fitz_doc:
                     fitz_doc.close()
                
                if fast_fitz_text and len(fast_fitz_text) > 200:
                    logging.info(
                        f"[{pdf_filename} - Page {page.page_number}] multicolumn_fitz_fast selected (len={len(fast_fitz_text)}). Skipping pdfplumber sweep."
                    )
                    return fast_fitz_text.strip(), "multicolumn_fitz_fast"
            param_sets = [
                (True, 1, 1),
                (False, 3, 3),
                (True, 3, 3),
                (False, 1.5, 1.5),
                (True, 1.5, 1.5),
                (True, 5, 5),
            ]
            best_text_from_params: str | None = None
            best_params_info = "none"
            for layout_flag, xtol, ytol in param_sets:
                try:
                    cand = page.extract_text(
                        layout=layout_flag,
                        x_density=7.25,
                        y_density=13,
                        x_tolerance=xtol,
                        y_tolerance=ytol,
                        keep_blank_chars=False,
                    )
                except Exception as e:
                    logging.debug(f"[{pdf_filename} - Page {page.page_number}] page.extract_text failed (layout={layout_flag}, xtol={xtol}, ytol={ytol}): {e}")
                    continue
                if cand:
                    cand = cand.strip()
                    if not best_text_from_params or len(cand) > len(best_text_from_params):
                        best_text_from_params = cand
                        best_params_info = f"layout={'T' if layout_flag else 'F'}_xt{xtol}_yt{ytol}"
            text_to_return = best_text_from_params
            chosen_method_info = best_params_info

            is_multi_col_page = _is_likely_multicolumn(page, central_band_ratio=0.25)
            if is_multi_col_page:
                logging.info(f"[{pdf_filename} - Page {page.page_number}] Detected as likely multicolumn. Trying multicolumn extractors.")
                # Refactor fitz opening here as well - needs caller change
                fitz_doc = None
                if _FITZ_AVAILABLE:
                    try:
                        fitz_doc = fitz.open(page.pdf.stream.name)
                    except Exception as e:
                        logging.warning(f"[{pdf_filename} - Page {page.page_number}] Failed to open with fitz for multicolumn check: {e}")
                
                fitz_text = _safe_multicol_fitz(page, fitz_doc)
                if fitz_text and _better_multicol_text(fitz_text, text_to_return, rel_margin=0.02):
                    logging.debug(
                        f"[{pdf_filename} - Page {page.page_number}] multicolumn_fitz is better. Using it."
                    )
                    text_to_return = fitz_text
                    chosen_method_info = "multicolumn_fitz"
                else:
                    logging.debug(
                        f"[{pdf_filename} - Page {page.page_number}] multicolumn_fitz not better or no result. Trying other multicolumn methods."
                    )
                    multicol_v4 = _safe_multicol_v4(page)
                    if multicol_v4 and _better_multicol_text(multicol_v4, text_to_return, rel_margin=0.02):
                        multicol_text = multicol_v4
                        multicol_method = "multicolumn_v4"
                    else:
                        multicol_v3 = _safe_multicol_v3(page)
                        if multicol_v3 and _better_multicol_text(multicol_v3, text_to_return, rel_margin=0.02):
                            multicol_text = multicol_v3
                            multicol_method = "multicolumn_v3"
                        else:
                            multicol_v2 = _extract_text_multicolumn_v2(page)
                            if multicol_v2 and _better_multicol_text(multicol_v2, text_to_return, rel_margin=0.02):
                                multicol_text = multicol_v2
                                multicol_method = "multicolumn_v2"
                            else:
                                multicol_simple = _extract_text_multicolumn(page)
                                if multicol_simple and _better_multicol_text(multicol_simple, text_to_return, rel_margin=0.02):
                                    multicol_text = multicol_simple
                                    multicol_method = "multicolumn_simple"
                    if multicol_text: # v4,v3,v2,simple のいずれかが採用された場合
                        logging.debug(
                            f"[{pdf_filename} - Page {page.page_number}] {multicol_method} better than params by >2% (effective). Using {multicol_method}."
                        )
                        text_to_return = multicol_text
                        chosen_method_info = multicol_method
                    else:
                        logging.debug(
                            f"[{pdf_filename} - Page {page.page_number}] No other multicolumn extractor produced a better result. Keeping params:({chosen_method_info})."
                        )
                    if fitz_doc: # Close fitz_doc if opened
                        fitz_doc.close()

            # Preemptive fitz check - needs caller change for doc object
            pre_fitz_text = None
            if _FITZ_AVAILABLE:
                 fitz_doc = None
                 try:
                     fitz_doc = fitz.open(page.pdf.stream.name)
                     pre_fitz_text = _safe_multicol_fitz(page, fitz_doc)
                 except Exception as e:
                     logging.warning(f"[{pdf_filename} - Page {page.page_number}] Failed to open with fitz for preemptive check: {e}")
                 finally:
                     if fitz_doc:
                         fitz_doc.close()

            if pre_fitz_text and _better_multicol_text(pre_fitz_text, text_to_return, rel_margin=0.02):
                logging.debug(
                    f"[{pdf_filename} - Page {page.page_number}] preemptive multicolumn_fitz is better (len={len(pre_fitz_text)} vs {len(text_to_return or '')}). Adopting it."
                )
                text_to_return = pre_fitz_text
                chosen_method_info = "multicolumn_fitz_preemptive"

            logging.info( 
                f"[{pdf_filename} - Page {page.page_number}] Final choice: Method='{chosen_method_info}', Length={len(text_to_return or ''),}, (WasLikelyMulticolumn={is_multi_col_page})"
            )

        if text_to_return:
            text_to_return = text_to_return.strip()
            logging.debug(f"[{pdf_filename} - Page {page.page_number}] pdfplumber extracted (patent_mode={patent_mode}, method={chosen_method_info}): {text_to_return[:100]}...")
            return text_to_return, chosen_method_info
        else:
            logging.debug(f"[{pdf_filename} - Page {page.page_number}] pdfplumber no text (patent_mode={patent_mode})")
            return None, chosen_method_info
    except Exception as e:
        logging.warning(f"[{pdf_filename} - Page {page.page_number}] _extract_text_with_pdfplumber error: {e}", exc_info=True)
        return None, f"error_in_pdfplumber_extraction: {e}"

def _extract_text_with_ocr(
    pdf_path: Path,
    page_num: int,
    ocr_lang: str = "jpn",
    rotate_degrees: int = 0,
    psm: int = 6,
) -> Optional[str]:
    """pdf2image + Tesseract OCR でページを認識し文字列を返す。

    Args:
        pdf_path: 処理する PDF ファイル
        page_num: 1-indexed ページ番号
        ocr_lang: Tesseract 言語データ (例: 'jpn' / 'jpn_vert')
        rotate_degrees: 画像を時計回り回転させる角度 (0/90/180/270)
        psm: page segmentation mode (Tesseract --psm)
    """
    logging.info(f"OCR処理を開始 (ページ {page_num}, lang={ocr_lang}, rot={rotate_degrees}) ...")

    # 言語フォールバックリストを構築
    langs_to_try = [ocr_lang]
    if ocr_lang == "jpn_vert" and "jpn" not in langs_to_try:
        langs_to_try.append("jpn")

    try:
        images = convert_from_path(
            pdf_path,
            first_page=page_num,
            last_page=page_num,
            fmt="png",
            thread_count=1,
            dpi=400,  # DPI を引き上げ
        )
        if not images:
            logging.warning(f"pdf2imageで画像への変換に失敗 (ページ {page_num})")
            return None

        img = images[0]
        if rotate_degrees:
            img = img.rotate(rotate_degrees, expand=True)

        # 画像前処理 (傾き補正・二値化等)
        img = _preprocess_image_for_ocr(img)

        for lang in langs_to_try:
            try:
                tess_cfg = f"--psm {psm}"
                logging.info(f"  [OCR] Trying Tesseract with lang='{lang}' ...")
                text = pytesseract.image_to_string(img, lang=lang, config=tess_cfg)
                if text and text.strip():
                    logging.info(f"OCR処理が完了 (ページ {page_num}, lang={lang})")
                    return text.strip()
                else:
                    logging.warning(f"  [OCR] No text returned for lang='{lang}'.")
            except pytesseract.TesseractError as te:
                logging.warning(f"  [OCR] TesseractError with lang='{lang}': {te}. 試行を続行します。")
                continue

        # 全ての言語で失敗
        logging.warning(f"OCR全試行でテキスト取得に失敗しました (ページ {page_num})")
        return None

    except (PDFPageCountError, PDF2ImageSyntaxError) as e:
        logging.error(f"pdf2imageでの画像変換中にエラー (ページ {page_num}): {e}", exc_info=True)
        if "poppler" in str(e).lower() or isinstance(e, PDFPageCountError): # PDFPageCountError も Poppler 起因の場合がある
            logging.error("Popplerがインストールされていないか、PATHが通っていない可能性があります。")
            logging.error("Windowsでは、Popplerのバイナリをダウンロードし、poppler/bin をPATHに追加してください。")
            logging.error("macOSでは、`brew install poppler` でインストールできます。")
            logging.error("Linuxでは、`sudo apt-get install poppler-utils` (Debian/Ubuntu) や `sudo yum install poppler-utils` (Fedora) などでインストールできます。")
        return None
    except PDFInfoNotInstalledError as e:
        logging.error(f"Poppler (pdfinfo) が見つかりません (ページ {page_num}): {e}", exc_info=True)
        logging.error("Popplerがインストールされていないか、PATHが通っていない可能性があります。")
        logging.error("Windowsでは、Popplerのバイナリをダウンロードし、poppler/bin をPATHに追加してください。")
        logging.error("macOSでは、`brew install poppler` でインストールできます。")
        logging.error("Linuxでは、`sudo apt-get install poppler-utils` (Debian/Ubuntu) や `sudo yum install poppler-utils` (Fedora) などでインストールできます。")
        return None
    except pytesseract.TesseractNotFoundError:
        logging.error("Tesseract OCRが見つかりません。インストールされているか、PATHを確認してください。", exc_info=True)
        return None
    except Exception as e:
        logging.error(f"OCR処理中に予期せぬエラー (ページ {page_num}): {e}", exc_info=True)
        return None

def _split_text_into_paragraphs(text: str, source: str = "unknown") -> List[Tuple[str, List[str]]]:
    """抽出されたテキストを段落に分割・整形し、タグを付与する"""
    
    # 前処理: ヘッダー/フッター除去
    text = _clean_extracted_text(text)

    paragraphs_with_tags: List[Tuple[str, List[str]]] = []
    current_paragraph_lines: list[str] = []

    # 句読点の直後改行 → そのまま残しつつ、改行2連へ置換して段落区切り強調
    text = re.sub(r"([。])\n", r"\1\n\n", text)

    # 行単位に処理
    prev_line_text: str | None = None
    for line in text.split("\n"):
        stripped_line = line.strip()

        # 同一行が連続する場合はスキップ (PDF の抽出で重複することがある)
        if prev_line_text is not None and stripped_line == prev_line_text:
            continue

        prev_line_text = stripped_line

        if stripped_line:
            current_paragraph_lines.append(stripped_line)
        else:
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

    # 段落の重複除去 (全文一致で連続する場合のみ)
    deduped_paragraphs: List[Tuple[str, List[str]]] = []
    for p_text, p_tags in paragraphs_with_tags:
        if deduped_paragraphs and deduped_paragraphs[-1][0] == p_text:
            continue
        deduped_paragraphs.append((p_text, p_tags))

    filtered_paragraphs = [
        (p_text, p_tags) for p_text, p_tags in deduped_paragraphs if MIN_LEN <= len(p_text) <= MAX_LEN
    ]

    if not filtered_paragraphs and paragraphs_with_tags:
         logging.debug(f"分割後の段落あり、ただし文字数フィルタで全て除外 (元段落数: {len(paragraphs_with_tags)}, ソース: {source})")
    elif not paragraphs_with_tags:
         logging.debug(f"テキストから段落への分割結果が0件 (ソース: {source})")

    return filtered_paragraphs

# -------------------------------------------------
# 追加: マルチカラムページ用テキスト再構成ヘルパ
# -------------------------------------------------

def _extract_text_multicolumn(page: Page) -> Optional[str]:
    """非常に単純な2段組レイアウト復元ヒューリスティック

    • ページの幅でおおよその列境界 (中央) を決め、左列 → 右列 の順で縦書き（top順）に結合。
    • 行単位のグルーピングは word['top'] を四捨五入して近似。同一Y座標でソート。

    厳密ではないが、pdfplumber の layout=True が行順を乱すケースで有効なことがある。
    """
    pdf_filename = Path(page.pdf.stream.name).name # Get filename for logging
    logging.debug(f"[{pdf_filename} - Page {page.page_number}] _extract_text_multicolumn: Start. W={page.width:.2f}, H={page.height:.2f}")

    try:
        words = page.extract_words(x_tolerance=1.5, y_tolerance=1.5, keep_blank_chars=False)
    except Exception as e:
        logging.debug(f"[{pdf_filename} - Page {page.page_number}] _extract_text_multicolumn: extract_words failed: {e}")
        return None

    if not words:
        logging.debug(f"[{pdf_filename} - Page {page.page_number}] _extract_text_multicolumn: No words extracted.")
        return None

    mid_x = page.width * 0.5
    logging.debug(f"[{pdf_filename} - Page {page.page_number}] _extract_text_multicolumn: mid_x={mid_x:.2f}")

    from collections import defaultdict
    left_lines: "defaultdict[float, list]" = defaultdict(list)
    right_lines: "defaultdict[float, list]" = defaultdict(list)

    for w_idx, w in enumerate(words):
        line_key = round(w["top"] / 2) * 2
        target_column = "left" if w["x0"] < mid_x else "right"
        target_dict = left_lines if target_column == "left" else right_lines
        target_dict[line_key].append(w)
        # Limit verbose word logging to avoid flooding
        if w_idx < 3 or (len(words) - w_idx) < 3 : 
             logging.debug(f"  [{pdf_filename} - Page {page.page_number} Word {w_idx}]: '{w['text'][:20]}' (x0:{w['x0']:.1f}, top:{w['top']:.1f}) -> key:{line_key}, col:{target_column}")

    def assemble(lines_dict, column_name: str):
        assembled_lines: list[str] = []
        logging.debug(f"[{pdf_filename} - Page {page.page_number}] _extract_text_multicolumn: Assembling {column_name} (keys: {len(lines_dict)})")
        sorted_line_keys = sorted(lines_dict.keys())
        for i, y_key in enumerate(sorted_line_keys):
            line_words = sorted(lines_dict[y_key], key=lambda x: x["x0"])
            current_line_text = " ".join(w["text"] for w in line_words)
            assembled_lines.append(current_line_text)
            # Limit verbose line logging
            if i < 2 or (len(sorted_line_keys) - i) < 2: 
                logging.debug(f"    [{pdf_filename} - Page {page.page_number} {column_name} line {i} (y:{y_key:.1f})]: '{current_line_text[:70]}'")
        return "\n".join(assembled_lines)

    left_text = assemble(left_lines, "left")
    right_text = assemble(right_lines, "right")

    logging.debug(f"[{pdf_filename} - Page {page.page_number}] _extract_text_multicolumn: Left text (first 100):\\n'{left_text[:100]}'")
    logging.debug(f"[{pdf_filename} - Page {page.page_number}] _extract_text_multicolumn: Right text (first 100):\\n'{right_text[:100]}'")

    if not left_text and not right_text:
        logging.debug(f"[{pdf_filename} - Page {page.page_number}] _extract_text_multicolumn: Both texts empty.")
        return None

    combined = (left_text + "\n" + right_text).strip()
    # logging.debug(f"[{pdf_filename} - Page {page.page_number}] _extract_text_multicolumn: Combined before re.sub (first 100):\\n'{combined[:100]}'")
    combined = re.sub(r"\n{3,}", "\n\n", combined)
    logging.debug(f"[{pdf_filename} - Page {page.page_number}] _extract_text_multicolumn: Final combined (first 100):\\n'{combined[:100]}'")
    return combined if combined else None

# -------------------------------------------------
# 追加: 動的クラスタリング版マルチカラム抽出 (v2)
# -------------------------------------------------

def _extract_text_multicolumn_v2(page: Page) -> Optional[str]:
    """改良版 2 カラムテキスト抽出

    1. すべての単語を抽出し、その中心 X 座標の分布を 2 クラスタに分割して列境界を推定。
       ・中央値より左 / 右で粗い初期クラスタ分け → 平均を取り左右クラスタ中心を決定。
    2. 中心の中点を境界線として左右振り分け。
    3. 行クラスタは numpy で top 座標をクラスタリング (近接融合)。
    4. 左列を上→下、右列を上→下で結合し返す。
    """
    from collections import defaultdict
    import numpy as np

    pdf_filename = Path(page.pdf.stream.name).name

    try:
        words = page.extract_words(x_tolerance=1.5, y_tolerance=1.5, keep_blank_chars=False)
    except Exception as e:
        logging.debug(f"[{pdf_filename} - Page {page.page_number}] _extract_text_multicolumn_v2: extract_words failed: {e}")
        return None

    if not words:
        return None

    # 1. 列クラスタリング (簡易 1D k-means, k=2)
    x_centers = np.array([(w["x0"] + w["x1"]) / 2 for w in words])

    # 初期センタ: 四分位数 25%, 75%
    c_left, c_right = np.percentile(x_centers, [25, 75])

    for _ in range(10):
        # 代入ステップ
        dist_left = np.abs(x_centers - c_left)
        dist_right = np.abs(x_centers - c_right)
        assign_left = dist_left <= dist_right

        if assign_left.all() or (~assign_left).all():
            break  # すべて同じクラスタになった → 不適

        # 更新ステップ
        new_c_left = x_centers[assign_left].mean()
        new_c_right = x_centers[~assign_left].mean()

        # 収束チェック
        if np.isclose(new_c_left, c_left) and np.isclose(new_c_right, c_right):
            break
        c_left, c_right = new_c_left, new_c_right

    center_left, center_right = sorted([c_left, c_right])

    if center_right - center_left < 50:
        return None

    boundary_x = (center_left + center_right) / 2
    logging.debug(
        f"[{pdf_filename} - Page {page.page_number}] _extract_text_multicolumn_v2: boundary_x={boundary_x:.2f} (centers {center_left:.1f}/{center_right:.1f})"
    )

    # 2. 単語の振り分け
    left_lines: "defaultdict[float, list]" = defaultdict(list)
    right_lines: "defaultdict[float, list]" = defaultdict(list)

    for w in words:
        tgt = left_lines if ((w["x0"] + w["x1"]) / 2) < boundary_x else right_lines
        # 行キー: 近接 3pt ごとにまとめる
        key = round(w["top"] / 3) * 3
        tgt[key].append(w)

    def assemble(col_dict: dict[float, list]):
        lines_text: list[str] = []
        for y in sorted(col_dict.keys()):
            sorted_words = sorted(col_dict[y], key=lambda _w: _w["x0"])
            line = " ".join(w["text"] for w in sorted_words)
            lines_text.append(line)
        return "\n".join(lines_text).strip()

    left_text = assemble(left_lines)
    right_text = assemble(right_lines)

    if not left_text and not right_text:
        return None

    combined = (left_text + "\n" + right_text).strip()
    combined = re.sub(r"\n{3,}", "\n\n", combined)
    return combined if combined else None

# -------------------------------------------------
# 追加: 行ベース境界推定版マルチカラム抽出 (v3)
# -------------------------------------------------

def _better_multicol_text(candidate: str, baseline: str | None, rel_margin: float = 0.02) -> bool:
    """候補テキストが baseline より有用かを多角的に評価するヒューリスティック。

    1. 実効長（連続スペース圧縮後の長さ）
    2. ユニーク語数（日本語・英語を簡易抽出）
    3. 重複度 (= 実効長 / ユニーク語数)

    • ① or ② が *rel_margin* 以上長い／多いなら採用。
    • 一方で実効長が少し短くても、重複度が baseline より高い (= 重複が少ない) 場合も採用。
    """

    if not candidate:
        return False

    if baseline is None:
        return True

    # 正規化
    cand_norm = _normalize_text_for_compare(candidate)
    base_norm = _normalize_text_for_compare(baseline)

    eff_c, eff_b = len(cand_norm), len(base_norm)

    word_re = re.compile(r"[\w\u3040-\u30FF\u4E00-\u9FFF]+")
    words_c = word_re.findall(cand_norm)
    words_b = word_re.findall(base_norm)

    wc_c, wc_b = len(words_c), len(words_b)

    # ① 文字長優位
    if eff_c > eff_b * (1 + rel_margin):
        return True

    # ② 単語数優位
    if wc_c > wc_b * (1 + rel_margin):
        return True

    # ③ 重複度（低いほど良い）
    if wc_c == 0 or wc_b == 0:
        return False
    dup_c = eff_c / wc_c
    dup_b = eff_b / wc_b

    # baseline の重複度が candidate の 10% 以上大きい かつ
    # 文字長が baseline の 85% 以上である場合のみ candidate を採用。
    if dup_b > dup_c * 1.10 and eff_c >= eff_b * 0.85:
        return True

    return False

def _extract_text_multicolumn_v3(page: Page) -> Optional[str]:
    """行レベルで境界を推定する改良版 2〜3段組抽出

    1. extract_words で word を取得し、Y 位置で行クラスタリング。
    2. 行の x_center を使って列分割境界を決定。
       ・x_center を昇順ソートし、隣接差分の最大値を境界候補とする。
       ・差分がページ幅の 8 % 以上あれば 2 カラムとみなす。
    3. 左列→右列の順で行を結合。
    """
    import numpy as np
    from collections import defaultdict
    pdf_filename = Path(page.pdf.stream.name).name

    try:
        words = page.extract_words(x_tolerance=1.5, y_tolerance=1.5, keep_blank_chars=False)
    except Exception as e:
        logging.debug(f"[{pdf_filename} - Page {page.page_number}] _extract_text_multicolumn_v3: extract_words failed: {e}")
        return None

    if not words:
        return None

    line_dict: defaultdict[float, list] = defaultdict(list)
    for w in words:
        try:
            key = round(float(w["top"]) / 3) * 3  # 3pt 単位で行を近似
        except Exception:
            continue
        line_dict[key].append(w)

    if len(line_dict) < 10:
        return None

    lines = []
    x_centers = []
    for y in sorted(line_dict.keys()):
        lwords = sorted(line_dict[y], key=lambda _w: _w["x0"])
        text = " ".join(w["text"] for w in lwords)
        if not text.strip():
            continue
        x_center = np.mean([(w["x0"] + w["x1"]) / 2 for w in lwords])
        x_centers.append(x_center)
        lines.append({"y": y, "text": text, "x_center": x_center})

    if len(lines) < 8:
        return None

    # --- K-Means 1D (k=2) for line centers ---
    c_left, c_right = np.percentile(x_centers, [25, 75])
    for _ in range(8):
        dist_l = np.abs(x_centers - c_left)
        dist_r = np.abs(x_centers - c_right)
        assign_l = dist_l <= dist_r
        if assign_l.all() or (~assign_l).all():
            break
        new_c_left = x_centers[assign_l].mean()
        new_c_right = x_centers[~assign_l].mean()
        if np.isclose(new_c_left, c_left) and np.isclose(new_c_right, c_right):
            break
        c_left, c_right = new_c_left, new_c_right

    if abs(c_right - c_left) < page.width * 0.15:
        return None

    boundary_x = (c_left + c_right) / 2
    left_lines = [ln for ln in lines if ln["x_center"] < boundary_x]
    right_lines = [ln for ln in lines if ln["x_center"] >= boundary_x]

    if len(left_lines) < 3 or len(right_lines) < 3:
        return None

    left_lines_sorted = sorted(left_lines, key=lambda l: l["y"])
    right_lines_sorted = sorted(right_lines, key=lambda l: l["y"])

    left_text = "\n".join(l["text"] for l in left_lines_sorted)
    right_text = "\n".join(l["text"] for l in right_lines_sorted)

    combined = (left_text + "\n" + right_text).strip()
    combined = re.sub(r"\n{3,}", "\n\n", combined)
    logging.debug(
        f"[{pdf_filename} - Page {page.page_number}] _extract_text_multicolumn_v3: boundary_x={boundary_x:.1f}, len_left={len(left_text)}, len_right={len(right_text)}, centers=({c_left:.1f},{c_right:.1f})"
    )
    return combined if combined else None

# -------------------------------------------------
# 追加: 行内ギャップ分割版マルチカラム抽出 (v4)
# -------------------------------------------------

def _extract_text_multicolumn_v4(page: Page) -> Optional[str]:
    """行内の最⼤ギャップで 2 カラムを復元する方式

    • 行クラスタリング (Y 近接 3pt)
    • 各行で word をソートし、隣接 word 間の X ギャップを取得
      ‑ 最大ギャップがページ幅 8 % 以上 → その行は 2 カラムと判断し分割
      ‑ そうでなければ 1 カラム行として左カラム側に寄せる
    • 左列→右列の順で上→下に結合
    """
    from collections import defaultdict
    import numpy as np

    pdf_filename = Path(page.pdf.stream.name).name

    try:
        words = page.extract_words(x_tolerance=1.5, y_tolerance=1.5, keep_blank_chars=False)
    except Exception as e:
        logging.debug(f"[{pdf_filename} - Page {page.page_number}] _extract_text_multicolumn_v4: extract_words failed: {e}")
        return None

    if not words:
        return None

    # 行クラスタ
    line_dict: defaultdict[float, list] = defaultdict(list)
    for w in words:
        try:
            key = round(float(w["top"]) / 3) * 3
        except Exception:
            continue
        line_dict[key].append(w)

    if len(line_dict) < 10:
        return None

    left_lines, right_lines = [], []
    page_width = page.width
    for y in sorted(line_dict.keys()):
        lwords = sorted(line_dict[y], key=lambda _w: _w["x0"])
        if not lwords:
            continue
        # 最大ギャップ検出
        gaps = np.diff([w["x0"] for w in lwords])
        if gaps.size == 0:
            left_lines.append(" ".join(w["text"] for w in lwords))
            continue
        max_gap = gaps.max()
        if max_gap < page_width * 0.08:
            # 1 カラム行
            left_lines.append(" ".join(w["text"] for w in lwords))
            continue

        # 2 カラム行として分割
        split_idx = int(np.argmax(gaps)) + 1
        left_part = " ".join(w["text"] for w in lwords[:split_idx])
        right_part = " ".join(w["text"] for w in lwords[split_idx:])
        left_lines.append(left_part)
        right_lines.append(right_part)

    if not left_lines and not right_lines:
        return None

    combined = ("\n".join(left_lines) + "\n" + "\n".join(right_lines)).strip()
    combined = _normalize_text_for_compare(combined)
    logging.debug(
        f"[{pdf_filename} - Page {page.page_number}] _extract_text_multicolumn_v4: lines L={len(left_lines)}, R={len(right_lines)}"
    )
    return combined if combined else None

# --------------------------
# 安全ラッパー
# --------------------------

def _safe_multicol_v3(page: Page) -> Optional[str]:
    try:
        return _extract_text_multicolumn_v3(page)
    except Exception as e:
        logging.debug(
            f"[{Path(page.pdf.stream.name).name} - Page {page.page_number}] _extract_text_multicolumn_v3 exception: {e}"
        )
        return None

def _safe_multicol_v4(page: Page) -> Optional[str]:
    try:
        return _extract_text_multicolumn_v4(page)
    except Exception as e:
        logging.debug(
            f"[{Path(page.pdf.stream.name).name} - Page {page.page_number}] _extract_text_multicolumn_v4 exception: {e}"
        )
        return None

# 新規: PyMuPDF 版の安全ラッパ
def _safe_multicol_fitz(page: Page, fitz_doc: Optional[fitz.Document]) -> Optional[str]:
    logging.debug(f"In _safe_multicol_fitz, _FITZ_AVAILABLE is: {_FITZ_AVAILABLE}")
    if not _FITZ_AVAILABLE or not fitz_doc:
        return None
    try:
        return _extract_text_multicolumn_fitz(page, fitz_doc)
    except Exception as e:
        logging.debug(
            f"[{Path(page.pdf.stream.name).name} - Page {page.page_number}] _extract_text_multicolumn_fitz exception: {e}", exc_info=True
        )
        return None

# -------------------------------------------------
# 追加: PyMuPDF ベースのカラム抽出 (fitz)
# -------------------------------------------------

# try...except ImportError...else ブロックは削除し、関数定義を直接記述する
# （_FITZ_AVAILABLE フラグで事前チェックするため）

def _extract_text_multicolumn_fitz(page: Page, fitz_doc: Optional[fitz.Document]) -> Optional[str]:
    """PyMuPDF (fitz) を使用して、テキストブロックをベースに複数カラムのテキストを抽出する。
       fitz.Document オブジェクトを受け取るように変更。
    """
    if not _FITZ_AVAILABLE or not fitz_doc:
        if not _FITZ_AVAILABLE:
             logging.warning(f"[{Path(page.pdf.stream.name).name} - Page {page.page_number}] _extract_text_multicolumn_fitz called but fitz is not available.")
        if not fitz_doc:
             logging.warning(f"[{Path(page.pdf.stream.name).name} - Page {page.page_number}] _extract_text_multicolumn_fitz called without fitz_doc.")
        return None
    
    pdf_path = Path(page.pdf.stream.name)
    pdf_name = pdf_path.name
    doc_page_num = page.page_number -1 # 0-indexed
    logging.debug(f"[{pdf_name} - Page {page.page_number}] FITZ_DEBUG: Entering _extract_text_multicolumn_fitz.")

    try:
        # doc = fitz.open(pdf_path) # REMOVED - Use passed fitz_doc
        if doc_page_num >= len(fitz_doc):
             logging.error(f"[{pdf_name} - Page {page.page_number}] Invalid page number {doc_page_num} for fitz doc (len: {len(fitz_doc)})")
             return None
             
        logging.debug(f"[{pdf_name} - Page {page.page_number}] FITZ_DEBUG: Attempting to load page {doc_page_num} from fitz_doc...")
        p = fitz_doc[doc_page_num]
        page_width = p.rect.width
        page_height = p.rect.height
        logging.debug(f"[{pdf_name} - Page {page.page_number}] FITZ_DEBUG: Fitz page loaded. Page WxH: {page_width:.2f}x{page_height:.2f}. Attempting to get text blocks...")
        blocks = p.get_text("blocks", sort=True) # y-sorted blocks
        # doc.close() # REMOVED - Caller should close the document
        logging.debug(f"[{pdf_name} - Page {page.page_number}] FITZ_DEBUG: Initial {len(blocks)} blocks from get_text(\"blocks\", sort=True):")
        for b_idx, b_raw in enumerate(blocks):
            logging.debug(f"  FITZ_DEBUG: Raw Block {b_idx}: x0={b_raw[0]:.2f}, y0={b_raw[1]:.2f}, x1={b_raw[2]:.2f}, y1={b_raw[3]:.2f}, text='{b_raw[4][:30].replace('\\n', ' ')}...'")

    except Exception as e:
        logging.error(f"[{pdf_name} - Page {page.page_number}] FITZ_DEBUG: fitz get_text error: {e}", exc_info=True)
        return None

    if not blocks:
        logging.warning(f"[{pdf_name} - Page {page.page_number}] FITZ_DEBUG: No blocks found after get_text. Returning None.")
        return None

    # --- 1. カラム境界の推定 ---
    block_properties = [{'text': b[4], 'x0': b[0], 'y0': b[1], 'x1': b[2], 'y1': b[3], 'block_no': b[5]} for b in blocks]
    
    if not block_properties:
        logging.debug(f"[{pdf_name} - Page {page.page_number}] FITZ_DEBUG: No block_properties after initial processing.")
        logging.debug(f"[{pdf_path.name} - Page {page.page_number}] FITZ_DEBUG: No block_properties after initial processing.")
        return None
    
    x_boundaries = set()
    logging.debug(f"[{pdf_path.name} - Page {page.page_number}] FITZ_DEBUG: Processing {len(block_properties)} block_properties for X boundaries:")
    for b_idx, b_prop in enumerate(block_properties):
        logging.debug(f"  FITZ_DEBUG: BlockProp {b_idx}: x0={b_prop['x0']:.2f}, x1={b_prop['x1']:.2f}, text='{b_prop['text'][:30].replace('\\n', ' ')}...'")
        x_boundaries.add(b_prop['x0'])
        x_boundaries.add(b_prop['x1'])
    
    sorted_x = sorted(list(x_boundaries))
    sorted_x_str = ", ".join([f"{x:.2f}" for x in sorted_x])
    logging.debug(f"[{pdf_path.name} - Page {page.page_number}] FITZ_DEBUG: Sorted unique X boundaries ({len(sorted_x)}): [{sorted_x_str}]")
    
    potential_gutters = []
    if len(sorted_x) > 1:
        logging.debug(f"[{pdf_path.name} - Page {page.page_number}] FITZ_DEBUG: Identifying potential gutters...")
        for i in range(len(sorted_x) - 1):
            gap_start = sorted_x[i]
            gap_end = sorted_x[i+1]
            gap_width = gap_end - gap_start
            min_gap_abs = 5.0 
            min_gap_rel = page_width * 0.01 
            logging.debug(f"  FITZ_DEBUG: Gap candidate: {gap_start:.2f} - {gap_end:.2f} (width: {gap_width:.2f}). MinAbs: {min_gap_abs:.2f}, MinRel (1% of pgW): {min_gap_rel:.2f}")
            if gap_width > max(min_gap_rel, min_gap_abs):
                potential_gutters.append((gap_start, gap_end))
                logging.debug(f"    FITZ_DEBUG: -> Found potential gutter: ({gap_start:.2f}, {gap_end:.2f}), width={gap_width:.2f}")

    if not potential_gutters:
        logging.info(f"[{pdf_path.name} - Page {page.page_number}] FITZ_DEBUG: No significant gutters found. Assuming single column.")
        result_text = "\\n".join(b['text'].strip() for b in block_properties if b['text'].strip()).strip()
        logging.debug(f"[{pdf_path.name} - Page {page.page_number}] FITZ_DEBUG: Single column text (len {len(result_text)}): '{result_text[:100].replace('\\n', ' ')}...'")
        return result_text if result_text else None

    potential_gutters.sort(key=lambda g: (g[1] - g[0]), reverse=True)
    potential_gutters_str = ", ".join([f"({g[0]:.2f}-{g[1]:.2f}, w:{g[1]-g[0]:.2f})" for g in potential_gutters[:5]])
    logging.debug(f"[{pdf_path.name} - Page {page.page_number}] FITZ_DEBUG: Sorted potential gutters by width (desc): [{potential_gutters_str}]")

    selected_midpoints: List[float] = []
    min_gutter_pos_ratio = 0.15
    max_gutter_pos_ratio = 0.85
    min_separation = page_width * 0.30
    logging.debug(f"[{pdf_path.name} - Page {page.page_number}] FITZ_DEBUG: Selecting final column dividers. PageW: {page_width:.2f}, MinGutterPosRatio: {min_gutter_pos_ratio:.2f}, MaxGutterPosRatio: {max_gutter_pos_ratio:.2f}, MinSep: {min_separation:.2f}")

    for g_start, g_end in potential_gutters:
        mid_x = (g_start + g_end) / 2
        logging.debug(f"  FITZ_DEBUG: Considering gutter ({g_start:.2f}-{g_end:.2f}), mid_x: {mid_x:.2f}")
        if not (page_width * min_gutter_pos_ratio < mid_x < page_width * max_gutter_pos_ratio):
            logging.debug(f"    FITZ_DEBUG: -> Rejected: mid_x not in range ({page_width * min_gutter_pos_ratio:.2f} - {page_width * max_gutter_pos_ratio:.2f})")
            continue
        if any(abs(mid_x - prev) < min_separation for prev in selected_midpoints):
            logging.debug(f"    FITZ_DEBUG: -> Rejected: too close to already selected dividers {selected_midpoints}")
            continue
        selected_midpoints.append(mid_x)
        logging.debug(f"    FITZ_DEBUG: -> Accepted: mid_x {mid_x:.2f}. Current dividers: {selected_midpoints}")
        if len(selected_midpoints) >= 2: 
            logging.debug(f"  FITZ_DEBUG: Reached max 2 dividers (3 columns). Stopping.")
            break
    
    if not selected_midpoints and potential_gutters:
        best_gutter = potential_gutters[0]
        fallback_midpoint = (best_gutter[0] + best_gutter[1]) / 2
        logging.debug(f"[{pdf_path.name} - Page {page.page_number}] FITZ_DEBUG: No suitable dividers found with strict criteria. Fallback to widest gutter: ({best_gutter[0]:.2f}-{best_gutter[1]:.2f}), mid: {fallback_midpoint:.2f}")
        selected_midpoints.append(fallback_midpoint)

    column_dividers = sorted(selected_midpoints)
    column_dividers_str = ", ".join([f"{d:.2f}" for d in column_dividers])
    logging.debug(f"[{pdf_path.name} - Page {page.page_number}] FITZ_DEBUG: Final column dividers: [{column_dividers_str}]")

    num_columns = len(column_dividers) + 1
    logging.info(f"[{pdf_path.name} - Page {page.page_number}] FITZ_DEBUG: Estimated {num_columns} columns with dividers at [{column_dividers_str}]")

    # --- 2. ブロックをカラムに割り当て ---
    columns_data: List[List[Dict]] = [[] for _ in range(num_columns)]
    current_page_blocks = sorted(block_properties, key=lambda b: (b['y0'], b['x0'])) 
    logging.debug(f"[{pdf_path.name} - Page {page.page_number}] FITZ_DEBUG: Assigning {len(current_page_blocks)} blocks (sorted by y0, x0) to {num_columns} columns.")

    for block_idx, block in enumerate(current_page_blocks):
        block_center_x = (block['x0'] + block['x1']) / 2
        assigned_column = 0
        for i, divider_x in enumerate(column_dividers):
            if block_center_x > divider_x:
                assigned_column = i + 1
            else:
                break
        
        if assigned_column < num_columns:
             columns_data[assigned_column].append(block)
             logging.debug(f"  FITZ_DEBUG: Block {block_idx} (y0={block['y0']:.1f}, x0={block['x0']:.1f}, x1={block['x1']:.1f}, ctr_x={block_center_x:.2f}, text='{block['text'][:20].replace('\\n',' ')}...') -> Col {assigned_column}")
        else: 
             logging.warning(f"[{pdf_path.name} - Page {page.page_number}] FITZ_DEBUG: Block {block_idx} (center_x={block_center_x:.2f}) assigned to out-of-bounds column {assigned_column} (max={num_columns-1}). Assigning to last column.")
             columns_data[num_columns-1].append(block) 
    
    for c_idx, c_data in enumerate(columns_data):
        logging.debug(f"[{pdf_path.name} - Page {page.page_number}] FITZ_DEBUG: Column {c_idx} has {len(c_data)} blocks. Preview:")
        for b_idx_in_col, b_in_col in enumerate(c_data[:3]): 
            logging.debug(f"    FITZ_DEBUG: Col {c_idx} Block {b_idx_in_col}: y0={b_in_col['y0']:.2f}, x0={b_in_col['x0']:.2f}, text='{b_in_col['text'][:30].replace('\\n', ' ')}...'")

    # --- 3. 各カラム内でブロックをY座標でソートし、テキストを結合 ---
    result_text_parts = []
    logging.debug(f"[{pdf_path.name} - Page {page.page_number}] FITZ_DEBUG: Joining text for {num_columns} columns.")
    for i in range(num_columns):
        sorted_column_blocks = columns_data[i] 
        
        column_text_fragments = []
        for blk_idx, blk in enumerate(sorted_column_blocks):
            block_text_stripped = blk['text'].strip()
            if block_text_stripped:
                column_text_fragments.append(block_text_stripped)

        column_text = "\\n".join(column_text_fragments).strip()
        if column_text:
            logging.debug(f"  FITZ_DEBUG: Column {i} combined text (len={len(column_text)}): '{column_text[:100].replace('\\n', ' ')}...'")
            result_text_parts.append(column_text)
        else:
            logging.debug(f"  FITZ_DEBUG: Column {i} produced no text after stripping/joining.")
    
    final_text = "\\n\\n".join(result_text_parts).strip()

    logging.info(
        f"[{pdf_path.name} - Page {page.page_number}] FITZ_DEBUG: Extracted. Columns={num_columns}, Final text length={len(final_text)}"
    )
    logging.debug(f"[{pdf_path.name} - Page {page.page_number}] FITZ_DEBUG: Final text preview (first 200 chars): '{final_text[:200].replace('\\n', ' ')}...'")

    # --- 2.5 ゾーンベース結合 (単一カラムブロックとマルチカラムブロック混在対応) ---
    def _is_spanning(b: Dict) -> bool:
        """ブロックが全カラムを跨ぐ (フル幅) かどうかを判定するヘルパ"""
        if num_columns <= 1:
            return True  # 1カラムの場合は常にスパン扱い
        width = b['x1'] - b['x0']
        cross_cnt = sum(1 for div in column_dividers if b['x0'] < div < b['x1'])
        # 全てのカラム境界を跨ぎ、ページ幅の6割以上を占める場合のみフル幅とみなす
        return cross_cnt == len(column_dividers) and width >= page_width * 0.60

    has_spanning_block = any(_is_spanning(b) for b in block_properties) if num_columns > 1 else False
 
    if has_spanning_block:
        blocks_sorted_y = sorted(block_properties, key=lambda b: b['y0'])
        result_text_parts: List[str] = []
        idx_z = 0
        while idx_z < len(blocks_sorted_y):
            blk_z = blocks_sorted_y[idx_z]
            spans_columns = _is_spanning(blk_z)
            if spans_columns:
                txt_piece = blk_z['text'].strip()
                if txt_piece:
                    result_text_parts.append(txt_piece)
                idx_z += 1
                continue
            zone_blocks: List[Dict] = []
            while idx_z < len(blocks_sorted_y):
                zb = blocks_sorted_y[idx_z]
                zb_spans = _is_spanning(zb)
                if zb_spans:
                    break
                zone_blocks.append(zb)
                idx_z += 1
            if not zone_blocks:
                continue
            zone_cols: List[List[Dict]] = [[] for _ in range(num_columns)]
            for zb in zone_blocks:
                ctr_x = (zb['x0'] + zb['x1']) / 2
                cidx = 0
                for j, div in enumerate(column_dividers):
                    if ctr_x > div:
                        cidx = j + 1
                    else:
                        break
                zone_cols[cidx].append(zb)
            for c_blocks in zone_cols:
                if not c_blocks:
                    continue
                c_blocks_sorted = sorted(c_blocks, key=lambda b: b['y0'])
                c_text = "\n".join(b['text'].strip() for b in c_blocks_sorted if b['text'].strip()).strip()
                if c_text:
                    result_text_parts.append(c_text)
        final_zone_text = "\n\n".join(result_text_parts).strip()
        if final_zone_text:
            logging.info(f"[{pdf_path.name} - Page {page.page_number}] FITZ_DEBUG: Extracted with zone-aware assembly. Final text length={len(final_zone_text)}")
            logging.debug(f"[{pdf_path.name} - Page {page.page_number}] FITZ_DEBUG: Final zone text preview (first 200 chars): '{final_zone_text[:200].replace('\n', ' ')}...'")
            return final_zone_text

    # ゾーン方式を適用しない場合、または結果が得られなかった場合は列結合のみの結果を返す
    return final_text if final_text else None

def _extract_text_vertical_fitz(page: Page) -> Optional[str]:
    """PyMuPDF の span(wmode=1) 情報を用いた縦書きテキスト抽出。

    1. page.get_text("dict") でブロック → 行 → スパンを取得。
    2. wmode==1 (縦書き) スパンのみ対象。
    3. x0 位置で簡易クラスタリングしてカラムを推定 (右→左)。
    4. 各カラム内を y 昇順に並べてテキスト結合。
    5. (cid:xxx) 擬似文字を除去して返す。
    """
    if not _FITZ_AVAILABLE:
        logging.warning(f"[{Path(page.pdf.stream.name).name} - Page {page.page_number}] _extract_text_vertical_fitz called but fitz is not available.")
        return None
    
    pdf_path = Path(page.pdf.stream.name)
    try:
        logging.debug(f"[{pdf_path.name} - Page {page.page_number}] (vertical_fitz) Attempting to open with fitz: {pdf_path}")
        doc = fitz.open(pdf_path)
        logging.debug(f"[{pdf_path.name} - Page {page.page_number}] (vertical_fitz) Fitz doc opened. Loading page {page.page_number - 1}")
        p = doc[page.page_number - 1]
        page_dict = p.get_text("dict")
        page_width = p.rect.width
        doc.close()
        logging.debug(f"[{pdf_path.name} - Page {page.page_number}] (vertical_fitz) Fitz page_dict retrieved.")
    except Exception as e:
        logging.debug(f"[{pdf_path.name} - Page {page.page_number}] _extract_text_vertical_fitz error: {e}", exc_info=True)
        return None

    if not page_dict or "blocks" not in page_dict:
        return None

    spans: List[Dict] = []
    for blk in page_dict["blocks"]:
        for line in blk.get("lines", []):
            for sp in line.get("spans", []):
                if sp.get("wmode", 0) == 1 and sp.get("text"):
                    spans.append(sp)

    if not spans:
        return None

    # ---- カラム推定 ----
    col_threshold = max(page_width * 0.04, 18)  # 4% または18pt
    columns: List[Dict] = []  # each: {x: float, spans: List}

    # 右→左の順に処理したいので x0 降順でソート
    spans_sorted = sorted(spans, key=lambda s: (-s["bbox"][0], s["bbox"][1]))

    for sp in spans_sorted:
        x0 = sp["bbox"][0]
        assigned = False
        for col in columns:
            if abs(x0 - col["x"]) < col_threshold:
                col["spans"].append(sp)
                assigned = True
                break
        if not assigned:
            columns.append({"x": x0, "spans": [sp]})

    # 右→左の順で並び替え
    columns_sorted = sorted(columns, key=lambda c: -c["x"])

    result_cols: List[str] = []
    for col in columns_sorted:
        # y 座標昇順 (上→下)。bbox[1] が y0
        col_spans_sorted = sorted(col["spans"], key=lambda s: s["bbox"][1])
        texts_in_col: List[str] = []
        last_y = None
        for sp in col_spans_sorted:
            txt = sp["text"].replace("\n", "").strip()
            if not txt:
                continue
            # 同じ y 位置に複数スパンが来るケースでは改行しない
            cur_y = sp["bbox"][1]
            if last_y is not None and abs(cur_y - last_y) > 5:
                texts_in_col.append("\n")
            texts_in_col.append(txt)
            last_y = cur_y
        col_text = "".join(texts_in_col).strip()
        if col_text:
            result_cols.append(col_text)

    if not result_cols:
        return None

    final_text = "\n\n".join(result_cols)
    final_text = _remove_cid_artifacts(final_text)
    return final_text.strip() if final_text.strip() else None

# -------------------------------------------------
#  縦書き判定 & 抽出ユーティリティ
# -------------------------------------------------

def _is_likely_vertical_text(page: Page, sample_size: int = 200) -> bool:
    """ページが縦書きレイアウトである可能性を簡易判定する。

    ヒューリスティック:
    1. `extract_words` で取得した word の bbox を利用。
    2. 幅 (x1-x0) より高さ (bottom-top) が明らかに大きい word が多い場合を縦書きとみなす。
       ※回転 PDF など例外もあるため、ページメタ情報の `/Rotate` までは見ない簡易版。
    """

    try:
        words = page.extract_words(x_tolerance=3, y_tolerance=3, keep_blank_chars=False)
    except Exception:
        return False

    if not words:
        return False

    sample = words[:sample_size]
    vert_like = 0
    for w in sample:
        w_width = w["x1"] - w["x0"]
        w_height = w["bottom"] - w["top"]
        if w_height == 0:
            continue
        if w_height / max(w_width, 1) > 1.8:  # 高さが幅の1.8倍以上 → 縦長
            vert_like += 1

    return vert_like >= len(sample) * 0.3  # 30% 以上縦長なら縦書きと判断 (閾値緩和)

def _extract_text_vertical_pdfplumber(page: Page) -> Optional[str]:
    """pdfplumber の writing_mode='vertical' を用いた縦書きテキスト抽出"""
    try:
        txt = page.extract_text(layout=True, writing_mode="vertical", x_tolerance=3, y_tolerance=3)
    except Exception:
        return None

    return txt.strip() if txt else None

# 不正な (cid:123) のようなグリフ参照を取り除くヘルパ
_re_cid = re.compile(r"\\(cid:\\d+\\)")
# CJK 文字判定用の正規表現 (縦書き品質チェックなどで利用)
_re_cjk = re.compile(r"[\\u3000-\\u9fff]")
# ひらがなのみで構成される1-4文字の行にマッチする正規表現 (ルビ除去用)
_re_short_hiragana_line = re.compile(r"^[ぁ-ゞ]{1,4}$")

def _remove_cid_artifacts(text: str) -> str:
    """pdfplumber が Unicode マッピングに失敗した際に現れる (cid:123) 等を除去"""
    return _re_cid.sub("", text)

def _remove_short_hiragana_lines(text: str | None) -> str | None:
    """テキストから、ひらがなのみで構成される短い行（ルビと想定）を除去する。"""
    if not text:
        return None
    
    lines = text.splitlines()
    cleaned_lines = [line for line in lines if not _re_short_hiragana_line.fullmatch(line.strip())]
    return "\n".join(cleaned_lines)

def _extract_text_vertical_ocr(pdf_path: Path, page_num: int) -> Optional[str]:
    """縦書きページ用 OCR 抽出ラッパ。

    新ロジック:
    1. ページ画像を取得し前処理
    2. ページ幅中央で左右2カラムに分割
    3. 各カラムを90°回転させ、`jpn_vert` + `--psm 5` でOCR
    4. 文字数が少ない場合のみ psm=6 を試す
    5. 左右カラム結果を連結して返す
    """

    from PIL import Image  # ローカルインポート

    logging.info(f"[VerticalOCR] page {page_num}: begin column-wise OCR with jpn_vert (no rotation)")

    try:
        images = convert_from_path(
            pdf_path,
            first_page=page_num,
            last_page=page_num,
            fmt="png",
            dpi=400,
            thread_count=1,
        )
    except Exception as e:
        logging.error(f"[VerticalOCR] convert_from_path failed on page {page_num}: {e}")
        return None

    if not images:
        logging.warning(f"[VerticalOCR] No image returned for page {page_num}")
        return None

    page_img = images[0]

    # 前処理 (Deskew, 二値化など)
    page_img = _preprocess_image_for_ocr(page_img)

    w, h = page_img.size

    # -----------------------------
    # 列境界推定 (X-projection profile)
    # -----------------------------
    try:
        import cv2
        import numpy as np
    except ImportError as ie:
        logging.warning(f"[VerticalOCR] cv2/numpy import failed ({ie}). Fallback to 2-column split.")
        mid_x = w // 2
        gutters = [(mid_x-1, mid_x+1)]
    else:
        try:
            arr = np.array(page_img)
            if arr.ndim == 2:  # already grayscale
                gray = arr
            elif arr.shape[2] == 4:  # RGBA
                gray = cv2.cvtColor(arr, cv2.COLOR_RGBA2GRAY)
            else:  # assume RGB
                gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        except Exception as cv_err:
            logging.warning(f"[VerticalOCR] cvtColor failed ({cv_err}). Fallback to simple 2-column split.")
            mid_x = w // 2
            gutters = [(mid_x-1, mid_x+1)]
            gray = None  # prevent further cv2 processing

        if gray is not None:
            _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
            # invert (text = 1)
            bw_inv = 255 - bw
            proj = bw_inv.sum(axis=0) / 255  # black pixel counts per column
            white_thresh = h * 0.03  # 3% of height以下なら「白い列」
            is_white = proj < white_thresh
            gutters = []
            in_gap = False
            gap_start = 0
            for x, flag in enumerate(is_white):
                if flag and not in_gap:
                    in_gap = True
                    gap_start = x
                elif not flag and in_gap:
                    gap_end = x
                    if gap_end - gap_start >= 10:
                        gutters.append((gap_start, gap_end))
                    in_gap = False
            if in_gap:
                gap_end = w-1
                if gap_end - gap_start >= 10:
                    gutters.append((gap_start, gap_end))

            if not gutters:
                gutters = []

        # 列領域 = 以下の x 範囲
        boundaries = [0] + [int((g[0]+g[1])/2) for g in gutters] + [w]
        column_boxes = []
        for i in range(len(boundaries)-1):
            x0 = boundaries[i]
            x1 = boundaries[i+1]
            # スキップ極端に狭い列
            if x1 - x0 < 30:
                continue
            column_boxes.append((x0, 0, x1, h))

        # 右→左順にソート
        column_boxes.sort(key=lambda b: -b[2])

# -----------------------------------------------------------------------------
#  データベース関連 & メイン CLI  (復旧)
# -----------------------------------------------------------------------------

def extract_and_process_pdf(pdf_path: Path, patent_mode: bool=False) -> List[Tuple[str,int,int,str,str,List[str]]]:
    """PDF を開いて段落を抽出し返す。以前の高度版は削除されたため簡易実装を復旧。"""
    paragraphs: list[Tuple[str,int,int,str,str,List[str]]] = []
    pdf_name = pdf_path.name
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, 1):
                txt, method = _extract_text_with_pdfplumber(page, patent_mode)
                if not txt:
                    continue
                for idx, (p_text, tags) in enumerate(_split_text_into_paragraphs(txt, source=method)):
                    paragraphs.append((pdf_name, page_num, idx, p_text, method, tags))
    except Exception as e:
        logging.error(f"extract_and_process_pdf: {pdf_name} failed -> {e}")
    return paragraphs


# 公開シンボルを strategies に合わせて更新
__all__ = [
    "BaseExtractor",
    "SimpleExtractor",
    "AdvancedExtractor",
    "extract_and_process_pdf",
]