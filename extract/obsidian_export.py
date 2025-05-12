"""Obsidian 連携ユーティリティ

抽出した段落を YAML Front Matter 付き Markdown として Obsidian Vault に保存します。

現状の仕様:
1. `extract.config.OBSIDIAN_INBOX_PATH` が設定されていない場合は何もしない。
2. 段落ごとに個別ファイルを生成 (ファイル名: <doc_slug>_p<page>_<idx>.md)
3. 保存先ディレクトリは `<OBSIDIAN_INBOX_PATH>/zlr-snippets/` (存在しなければ作成)
4. YAML Front Matter:
   - id: <doc_id>-p<page>-<idx>
   - doc: <doc_id>
   - page: <page>
   - source: <source>
   - tags: [zlr, <タグ...>]
   - created: <タイムスタンプ>
5. 本文は引用ブロック (>) でスニペットを格納
"""

from __future__ import annotations

from pathlib import Path
from datetime import datetime
import re
import textwrap
import logging
from typing import List, Tuple

try:
    from .config import OBSIDIAN_INBOX_PATH
except Exception as e:  # pragma: no cover
    OBSIDIAN_INBOX_PATH = None  # type: ignore
    logging.warning(f"[ObsidianExport] config 読み込みに失敗しました: {e}. Obsidian 連携は無効化されます。")

# 型エイリアス (doc_id, page, idx, text, source, tags)
Paragraph = Tuple[str, int, int, str, str, List[str]]


def _slugify(text: str, max_len: int = 60) -> str:
    """ファイル名に使えない文字を `_` に置換し、長さを制限"""
    slug = re.sub(r"[^0-9A-Za-z-_]+", "_", text)
    if len(slug) > max_len:
        slug = slug[:max_len]
    return slug


def export_paragraphs_to_obsidian(paragraphs: List[Paragraph], *, target_dir: Path | None = None) -> None:
    """段落リストを Obsidian Vault に Markdown として保存する。

    Parameters
    ----------
    paragraphs : list
        (doc_id, page, idx, text, source, tags) のタプルのリスト
    target_dir : Path | None, optional
        保存先ディレクトリを明示的に指定する場合。省略時は
        OBSIDIAN_INBOX_PATH / 'zlr-snippets'
    """
    if OBSIDIAN_INBOX_PATH is None:
        logging.debug("[ObsidianExport] OBSIDIAN_INBOX_PATH が未設定のため、処理をスキップします。")
        return

    # デフォルト保存先
    if target_dir is None:
        target_dir = OBSIDIAN_INBOX_PATH / "zlr-snippets"

    try:
        target_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        logging.error(f"[ObsidianExport] 保存先ディレクトリの作成に失敗しました: {e}")
        return

    for doc_id, page, idx, text, source, tags in paragraphs:
        slug = _slugify(doc_id)
        file_name = f"{slug}_p{page:03d}_{idx:02d}.md"
        file_path = target_dir / file_name

        yaml_lines = [
            "---",
            f"id: {slug}-p{page}-{idx}",
            f"doc: {doc_id}",
            f"page: {page}",
            f"source: {source}",
            "tags: [zlr" + (", " + ", ".join(tags) if tags else "") + "]",
            f"created: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "---",
            "",
        ]

        body = textwrap.fill(text, width=80)
        content = "\n".join(yaml_lines) + "> " + body + "\n"

        try:
            with open(file_path, "w", encoding="utf-8") as fp:
                fp.write(content)
            logging.info(f"[ObsidianExport] 段落を保存しました: {file_path.relative_to(OBSIDIAN_INBOX_PATH)}")
        except Exception as e:
            logging.error(f"[ObsidianExport] ファイル書き込みに失敗しました ({file_path}): {e}") 