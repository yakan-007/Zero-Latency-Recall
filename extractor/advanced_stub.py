"""AdvancedExtractor (スタブ)

LayoutParser + PaddleOCR 等を用いた高度抽出ロジックを実装する予定のクラス。
現段階では ImportError を避けつつ、未実装を明示するみにマム実装に留める。"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

from .base import BaseExtractor


class AdvancedExtractor(BaseExtractor):
    """有償版で提供予定の高度 Extractor（未実装）。"""

    def ensure_requirements(self) -> None:  # noqa: D401
        # 依存ライブラリが存在するか軽く検査する例（今は警告のみ）
        try:
            import layoutparser  # type: ignore
            import paddleocr  # type: ignore
        except ImportError:
            # 依存が無い場合でもエラーにせず、process_pdf 内で明示エラーにする
            return

    def process_pdf(
        self, pdf_path: Path, **kwargs
    ) -> List[Tuple[str, int, int, str, str, List[str]]]:
        raise NotImplementedError(
            "AdvancedExtractor はまだ実装されていません。\n"
            "LayoutParser + PaddleOCR ベースの抽出ロジックを次フェーズで追加予定です。"
        ) 