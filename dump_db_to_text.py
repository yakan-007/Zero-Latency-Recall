#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SQLite データベースの内容をテキストファイルに書き出すスクリプト。
"""

import sqlite3
from pathlib import Path
import sys

# config から DB パスを読み込み (失敗しても動作継続)
try:
    from config import DB_PATH
except ImportError:
    print("警告: config.py が見つかりません。デフォルトのDBパス 'zlr.sqlite' を使用します。", file=sys.stderr)
    DB_PATH = Path(__file__).parent / "zlr.sqlite"

OUTPUT_FILE = Path("db_dump.txt")

def dump_database_to_text(db_path: Path, output_file: Path):
    """データベースの内容をテキストファイルに書き出す"""
    print(f"データベース '{db_path}' の内容を '{output_file}' に書き出します...")

    if not db_path.exists():
        print(f"エラー: データベースファイルが見つかりません: {db_path}", file=sys.stderr)
        return False

    row_count = 0
    try:
        with sqlite3.connect(db_path) as con, open(output_file, "w", encoding="utf-8") as f:
            # テーブルが存在するか確認
            cursor = con.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='docs'")
            if cursor.fetchone() is None:
                print(f"エラー: テーブル 'docs' がデータベース 'zlr.sqlite' に見つかりません。", file=sys.stderr)
                return False

            # tags カラムも含めて全データを取得
            # 旧スキーマとの互換性のため、PRAGMAでカラム情報を確認
            pragma_cur = con.execute("PRAGMA table_info(docs)")
            columns = [row[1] for row in pragma_cur.fetchall()]
            select_columns = "rowid, doc_id, page, text, tags" if "tags" in columns else "rowid, doc_id, page, text"

            cursor = con.execute(f"SELECT {select_columns} FROM docs ORDER BY doc_id, page")

            f.write(f"--- Database Dump ({db_path}) ---\n\n")
            for row in cursor:
                if "tags" in columns:
                    rowid, doc_id, page, text, tags_str = row
                else:
                    rowid, doc_id, page, text = row
                    tags_str = "[N/A]" # tagsカラムがない場合は N/A を表示
                # テキスト内の改行をスペースに置換して1行で表示（見やすさのため）
                formatted_text = ' '.join(text.splitlines()).strip()
                f.write(f"[{doc_id}] (Page: {page}) [ID:{rowid}] Tags: {tags_str}\n")
                f.write(f"{formatted_text}\n\n") # テキストの後に空行を入れる
                row_count += 1

            f.write(f"--- End of Dump ({row_count} rows) ---\n")

    except sqlite3.Error as e:
        print(f"データベースアクセス中にエラーが発生しました: {e}", file=sys.stderr)
        return False
    except OSError as e:
        print(f"ファイル書き込み中にエラーが発生しました ({output_file}): {e}", file=sys.stderr)
        return False
    except Exception as e:
        print(f"予期せぬエラーが発生しました: {e}", file=sys.stderr)
        return False

    print(f"書き出し完了。{row_count} 件のデータを '{output_file}' に保存しました。")
    return True

if __name__ == "__main__":
    if not dump_database_to_text(DB_PATH, OUTPUT_FILE):
        sys.exit(1) 