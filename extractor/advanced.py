"""AdvancedExtractor: LayoutParser + PaddleOCR 版

有償エディション向けの高精度抽出。
1. pdf2image でページを PIL.Image へ
2. LayoutParser Detectron2 モデル (PubLayNet) でテキストブロック検出
3. ブロックを crop → PaddleOCR (japan, cls=True) で OCR
4. 読み順 (x 좌표, y 좌標) に並べてテキスト結合
5. extract.py のユーティリティで段落分割して返す

依存: layoutparser[detectron2], paddleocr, pdf2image
"""

from __future__ import annotations

import importlib
import logging
from pathlib import Path
from typing import List, Tuple

from .base import BaseExtractor

# extract.py のユーティリティを再利用
extract_mod = importlib.import_module("extract")
_split_text_into_paragraphs = getattr(extract_mod, "_split_text_into_paragraphs")
_clean_extracted_text = getattr(extract_mod, "_clean_extracted_text")
_normalize_text_for_compare = getattr(extract_mod, "_normalize_text_for_compare")
_remove_cid_artifacts = getattr(extract_mod, "_remove_cid_artifacts")
_remove_short_hiragana_lines = getattr(extract_mod, "_remove_short_hiragana_lines")

logger = logging.getLogger(__name__)


class AdvancedExtractor(BaseExtractor):
    """LayoutParser + PaddleOCR ベースの実装。"""

    _layout_model = None  # class-level singleton (任意)
    _ocr_engine = None
    _pil_image = None  # PIL.Image 参照用
    _lp_module = None  # layoutparser 参照用 (無くても可)

    def __init__(self, patent_mode: bool = False) -> None:
        self.patent_mode = patent_mode
        self.ensure_requirements()

    # ------------------------------------------------------------------
    # 準備
    # ------------------------------------------------------------------

    def ensure_requirements(self) -> None:  # noqa: D401
        try:
            from paddleocr import PaddleOCR  # noqa: F401
            from pdf2image import convert_from_path  # noqa: F401
            from PIL import Image  # noqa: F401
            self._pil_image = Image
            # layoutparser は任意。インストールされていれば利用、無ければページ全体OCRフォールバック
            try:
                import layoutparser as lp  # type: ignore
                self._lp_module = lp
            except ImportError:
                self._lp_module = None
        except ImportError as e:  # pragma: no cover
            raise RuntimeError(
                "AdvancedExtractor を使用するには layoutparser[detectron2], paddleocr, pdf2image が必要です: "
                f"{e}"
            )

    # ------------------------------------------------------------------
    # 内部 helpers
    # ------------------------------------------------------------------

    @classmethod
    def _get_layout_model(cls):
        if cls._layout_model is None and cls._lp_module is not None and hasattr(cls._lp_module, "Detectron2LayoutModel"):
            logger.info("LayoutParser Detectron2 モデルをロード中 (PubLayNet)...")
            cls._layout_model = cls._lp_module.Detectron2LayoutModel(
                "lp://PubLayNet/faster_rcnn_R_50_FPN_3x/config",
                extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.5],
                label_map={0: "text", 1: "title", 2: "list", 3: "table", 4: "figure"},
            )
        return cls._layout_model  # None になる場合もある

    @classmethod
    def _get_ocr(cls):
        if cls._ocr_engine is None:
            from paddleocr import PaddleOCR

            logger.info("PaddleOCR エンジン (japan) を初期化中 ...")
            cls._ocr_engine = PaddleOCR(lang="japan", use_angle_cls=True, show_log=False)
        return cls._ocr_engine

    # ------------------------------------------------------------------

    def process_pdf(
        self, pdf_path: Path, **kwargs
    ) -> List[Tuple[str, int, int, str, str, List[str]]]:
        from pdf2image import convert_from_path

        force_ocr: bool = kwargs.get("force_ocr", False)

        # force_ocr が True の場合は LayoutParser をスキップ
        layout_model = None if force_ocr else self._get_layout_model()
        ocr = self._get_ocr()

        logger.info(f"AdvancedExtractor: '{pdf_path.name}' を処理開始")
        try:
            pages = convert_from_path(pdf_path, dpi=300, thread_count=2)
        except Exception as e:
            logger.error(f"pdf2image で {pdf_path.name} の変換に失敗: {e}")
            if "poppler" in str(e).lower() or "PDFPageCountError" in str(e) or "PDFInfoNotInstalledError" in str(e):
                logger.error("Popplerがインストールされていないか、PATHが通っていない可能性があります。")
                logger.error("macOSでは `brew install poppler`、Linuxでは `apt-get install poppler-utils` 等で導入ください。")
            return []

        output_paragraphs: List[Tuple[str, int, int, str, str, List[str]]] = []

        for page_idx, pil_img in enumerate(pages, 1):
            block_texts: List[str] = []
            if layout_model is not None:
                try:
                    layout = layout_model.detect(pil_img)
                    text_blocks = [b for b in layout if getattr(b, "type", "") == "text"]
                    logger.debug(f"Page {page_idx}: detected {len(text_blocks)} text blocks via layoutparser")

                    text_blocks.sort(key=lambda b: (b.block.y_1, b.block.x_1))  # type: ignore

                    for i, blk in enumerate(text_blocks):
                        x1, y1, x2, y2 = map(int, blk.coordinates)  # type: ignore
                        crop_img = pil_img.crop((x1, y1, x2, y2))
                        txt = self._ocr_image_crop(crop_img, ocr, page_idx, i)
                        if txt:
                            block_texts.append(txt)
                except Exception as lp_e:
                    logger.warning(f"Layout detection failed on page {page_idx}: {lp_e}. Fallback to full-page OCR.")

            # Fallback: Full page OCR if no blocks あるいは force_ocr 指定
            if force_ocr or not block_texts:
                logger.info(f"Page {page_idx}: Using full-page OCR fallback{' (force_ocr)' if force_ocr else ''}.")
                txt = self._ocr_image_crop(pil_img, ocr, page_idx, -1)
                if txt:
                    block_texts.append(txt)

            page_text = "\n\n".join(block_texts)
            if not page_text:
                logger.warning(f"Page {page_idx}: OCR でテキストを取得できませんでした")
                continue

            # ---- ポストプロセス ----
            page_text = _remove_cid_artifacts(page_text)
            page_text = _remove_short_hiragana_lines(page_text) or page_text
            page_text = _clean_extracted_text(page_text)

            para_with_tags = _split_text_into_paragraphs(page_text, source="advanced_layout_ocr")
            for para_idx, (ptext, tags) in enumerate(para_with_tags):
                output_paragraphs.append(
                    (pdf_path.name, page_idx, para_idx, ptext, "advanced_layout_ocr", tags)
                )

        logger.info(
            f"AdvancedExtractor: '{pdf_path.name}' 完了。抽出段落 {len(output_paragraphs)}"
        )
        return output_paragraphs

    def _ocr_image_crop(self, image, ocr, page_idx: int, blk_idx: int) -> str | None:
        """画像 (PIL.Image) を PaddleOCR に渡し、テキストを返すヘルパ"""
        try:
            # Pillow Image を NumPy 配列に明示的に変換
            import numpy as np
            img_np = np.array(image)
            result = ocr.ocr(img_np, cls=True)
            if not result or not result[0]:
                logger.debug(f"Page {page_idx} block {blk_idx}: PaddleOCR returned no result")
                return None
            lines = [r[1][0] for res_block in result for r in res_block]
            return "\n".join(lines).strip()
        except Exception as e:
            logger.exception(f"Page {page_idx} block {blk_idx}: PaddleOCR error caught")
            # --- フォールバック: pytesseract ---
            try:
                import pytesseract
                logger.info(
                    f"Page {page_idx} block {blk_idx}: PaddleOCR 失敗のため pytesseract にフォールバックします。"
                )
                # pytesseract は PIL.Image を直接受け取れる
                txt = pytesseract.image_to_string(image, lang="jpn")
                txt = txt.strip()
                if txt:
                    return txt
            except Exception as fallback_e:  # noqa: BLE001
                logger.warning(
                    f"Page {page_idx} block {blk_idx}: pytesseract フォールバックも失敗: {fallback_e}"
                )
            return None 