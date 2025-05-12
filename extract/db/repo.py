import sqlite3
import logging
from pathlib import Path
from typing import List, Tuple

# -----------------------------------------------------------------------------
#  データベース関連 & メイン CLI  (復旧) <- 元のコメントは不要かも
# -----------------------------------------------------------------------------

# 型エイリアス (doc_id, page, idx, text, source, [tags])
Paragraph = Tuple[str, int, int, str, str, List[str]]

def ensure_db_schema(db_path: Path) -> None:
    """SQLite DB に docs (FTS5) テーブルを用意します。既に存在する場合は何もしません。"""
    logging.info("[DB] ensure_db_schema: %s", db_path)
    with sqlite3.connect(db_path) as conn:
        cur = conn.cursor()
        cur.execute(
            """
            CREATE VIRTUAL TABLE IF NOT EXISTS docs USING fts5(
                text,               -- 検索対象本文
                doc_id UNINDEXED,   -- ファイル名など (検索対象外)
                page UNINDEXED,     -- ページ番号 (検索対象外)
                tags UNINDEXED,     -- 段落タグ (検索対象外)
                source UNINDEXED,   -- 取得ソース (pdfplumber/ocr 等)
                paragraph_num UNINDEXED,
                tokenize = 'trigram'
            )
            """
        )
        conn.commit()

def save_paragraphs_to_db(db_path: Path, doc_id: str, paragraphs: List[Paragraph]) -> None:
    """抽出した段落を docs テーブルへ一括保存します。"""
    if not paragraphs:
        logging.debug("[DB] save_paragraphs_to_db: 空の段落リストが渡されました。スキップします。")
        return

    with sqlite3.connect(db_path) as conn:
        cur = conn.cursor()
        cur.executemany(
            "INSERT INTO docs (text, doc_id, page, tags, source, paragraph_num) VALUES (?,?,?,?,?,?)",
            [
                (
                    txt,
                    doc_id,
                    page,
                    ",".join(tags) if tags else "",
                    source,
                    idx,
                )
                for (_, page, idx, txt, source, tags) in paragraphs
            ],
        )
        conn.commit()
        logging.info("[DB] %s に %d 段落を保存しました。", db_path.name, len(paragraphs)) 