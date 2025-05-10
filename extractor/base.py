"""共通抽象クラス BaseExtractor

この層では、PDF からテキストを抽出して段落化するまでの
共通インターフェイスのみを規定し、具象クラスは process_pdf を実装します。
"""

from __future__ import annotations

import abc
from pathlib import Path
from typing import List, Tuple


class BaseExtractor(abc.ABC):
    """Extractor の抽象基底。エディション毎に実装を切り替える。"""

    @abc.abstractmethod
    def process_pdf(
        self, pdf_path: Path, **kwargs
    ) -> List[Tuple[str, int, int, str, str, List[str]]]:
        """PDF を処理し、段落タプルのリストを返す。

        戻り値は extract.extract_and_process_pdf と互換にし、
        (filename, page_num, para_idx, text, source, tags) を返す。"""

    # ------------------------------------------------------------------
    # 以下は将来、共通ユーティリティ関数を置く予定。
    # 現段階では具象クラス側で extract.py の実装を呼び出すため空実装。
    # ------------------------------------------------------------------

    def ensure_requirements(self) -> None:  # noqa: D401
        """追加依存を動的 import したい場合にオーバーライドする。""" 