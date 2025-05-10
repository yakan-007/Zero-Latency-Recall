"""SimpleExtractor: 旧 pdfplumber ベースの実装をラップ

本実装では、既存の extract.py の extract_and_process_pdf() を呼び出すだけに留め、
将来的なリファクタリングフェーズでロジックをここへ移植する。"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

from .base import BaseExtractor

# 既存の関数を再利用
try:
    # ローカルパッケージ内ではなくルートの extract.py をインポート
    import importlib
    extract_mod = importlib.import_module("extract")
except ModuleNotFoundError:  # テスト時など extract.py がリネームされた場合
    extract_mod = None


class SimpleExtractor(BaseExtractor):
    """無償エディション向けの軽量 Extractor"""

    def __init__(self, patent_mode: bool = False):
        self.patent_mode = patent_mode

    def process_pdf(
        self, pdf_path: Path, **kwargs
    ) -> List[Tuple[str, int, int, str, str, List[str]]]:
        if extract_mod is None:
            raise RuntimeError("extract.py が見つからず SimpleExtractor を実行できません。")

        force_ocr: bool = kwargs.get("force_ocr", False)
        no_ocr_fallback: bool = kwargs.get("no_ocr_fallback", False)

        # 既存のメイン関数を呼び出す
        return extract_mod.extract_and_process_pdf(
            pdf_path,
            patent_mode=self.patent_mode,
            force_ocr=force_ocr,
            no_ocr_fallback=no_ocr_fallback,
        ) 