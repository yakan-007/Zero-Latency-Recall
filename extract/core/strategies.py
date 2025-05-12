"""Extractor 戦略クラス群 (拡張性重視)

BaseExtractor  : すべての抽出器の共通インターフェース。
SimpleExtractor: pdfplumber ベースの従来実装をラップ。
AdvancedExtractor: OCR / PyMuPDF など高度処理。外部実装が見つからない場合は SimpleExtractor 相当でフォールバック。

このモジュールを import すると、下記 3 クラスが公開されます。
"""

from __future__ import annotations

import abc
import logging
from pathlib import Path
from typing import List, Tuple

# ------------------------------------------------------------
# 型エイリアス
Paragraph = Tuple[str, int, int, str, str, List[str]]
# (doc_id, page, idx, text, source, [tags])
# ------------------------------------------------------------

class BaseExtractor(abc.ABC):
    """Extractor 抽象基底クラス"""

    def __init__(self, patent_mode: bool = False) -> None:  # noqa: D401
        self.patent_mode = patent_mode

    @abc.abstractmethod
    def process_pdf(self, pdf_path: Path, **kwargs) -> List[Paragraph]:  # pragma: no cover
        """1つの PDF を処理して段落リストを返す。"""
        raise NotImplementedError


class SimpleExtractor(BaseExtractor):
    """従来ロジック (pdfplumber) をそのまま呼び出す抽出器"""

    def process_pdf(self, pdf_path: Path, **kwargs) -> List[Paragraph]:
        # 循環 import 防止のためメソッド内で遅延 import
        from extract.core.extractor import extract_and_process_pdf  # local import
        return extract_and_process_pdf(pdf_path, patent_mode=self.patent_mode)


# 旧 AdvancedExtractor の外部実装が存在するか判定
try:
    from extractor.advanced import AdvancedExtractor as _LegacyAdvancedExtractor  # type: ignore
except Exception:  # noqa: BLE001
    _LegacyAdvancedExtractor = None


class AdvancedExtractor(BaseExtractor):
    """OCR 等を併用する高精度抽出器 (フォールバック付き)"""

    def __init__(self, patent_mode: bool = False) -> None:  # noqa: D401
        super().__init__(patent_mode=patent_mode)
        if _LegacyAdvancedExtractor is not None:
            self._impl = _LegacyAdvancedExtractor(patent_mode=patent_mode)  # type: ignore[arg-type]
        else:
            self._impl = None
            logging.warning(
                "[AdvancedExtractor] 外部実装が見つからないため SimpleExtractor 相当で動作します。"
            )

    def process_pdf(self, pdf_path: Path, force_ocr: bool = False, **kwargs) -> List[Paragraph]:
        if self._impl is not None:
            try:
                return self._impl.process_pdf(pdf_path, force_ocr=force_ocr, **kwargs)  # type: ignore[arg-type]
            except Exception as e:  # noqa: BLE001
                logging.error(
                    f"[AdvancedExtractor] 処理失敗 ({pdf_path.name}): {e}. SimpleExtractor にフォールバックします。"
                )
        # フォールバック (遅延 import)
        from extract.core.extractor import extract_and_process_pdf  # local import
        return extract_and_process_pdf(pdf_path, patent_mode=self.patent_mode)


__all__ = [
    "BaseExtractor",
    "SimpleExtractor",
    "AdvancedExtractor",
] 