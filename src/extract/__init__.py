"""extract パッケージ

外部から参照される主要シンボルを re-export します。
"""

from importlib import import_module

# 主要モジュールを遅延 import (循環防止)
_extractor = import_module("extract.core.extractor")

# エクスポートする属性
extract_and_process_pdf = getattr(_extractor, "extract_and_process_pdf")
TAG_DICT = getattr(_extractor, "TAG_DICT")
SimpleExtractor = getattr(_extractor, "SimpleExtractor")
AdvancedExtractor = getattr(_extractor, "AdvancedExtractor")
BaseExtractor = getattr(_extractor, "BaseExtractor")

__all__ = [
    "extract_and_process_pdf",
    "TAG_DICT",
    "SimpleExtractor",
    "AdvancedExtractor",
    "BaseExtractor",
] 