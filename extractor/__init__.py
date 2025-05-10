"""ZLR 抽出モジュール

このパッケージは、PDF からテキストを抽出するための複数の Extractor 実装を提供します。

- BaseExtractor: 共通インターフェイス
- SimpleExtractor: 既存の pdfplumber / Tesseract フォールバック方式
- AdvancedExtractor: LayoutParser + PaddleOCR 等を用いる高度版（有料エディション向け、現在はスタブ）
"""

from __future__ import annotations

from .base import BaseExtractor  # noqa: F401
from .simple import SimpleExtractor  # noqa: F401
from .advanced_stub import AdvancedExtractor as _StubAdvanced
try:
    from .advanced import AdvancedExtractor  # type: ignore # noqa: F401
except Exception:  # pragma: no cover
    # 高度版依存が入っていない環境ではスタブをエクスポート
    AdvancedExtractor = _StubAdvanced  # type: ignore

__all__ = [
    "BaseExtractor",
    "SimpleExtractor",
    "AdvancedExtractor",
] 