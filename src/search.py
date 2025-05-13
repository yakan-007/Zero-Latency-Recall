#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SQLite FTS5 テーブルを検索する CLI スクリプト
"""

import sqlite3
import sys
import argparse
import textwrap
import re
from pathlib import Path
import os # os モジュールをインポート (環境変数、パス操作用)
from datetime import datetime # 日時情報のためインポート

# config から DB パスと Obsidian パスを読み込み (失敗しても動作継続)
try:
    from extract.config import DB_PATH
except ImportError:
    print("警告: config.py が見つかりません。デフォルトのDBパス 'zlr.sqlite' を使用します。", file=sys.stderr)
    # プロジェクトルートからの相対パスを想定
    DB_PATH = Path(__file__).resolve().parent.parent / "zlr.sqlite" # search.py が src/ にあるので、親の親がプロジェクトルート

try:
    # OBSIDIAN_INBOX_PATH はオプション扱いなので、なくてもエラーにしない
    from extract.config import OBSIDIAN_INBOX_PATH
except (ImportError, AttributeError):
     OBSIDIAN_INBOX_PATH = None # 未設定の場合は None


def _build_where_clause(tokens: list[str]):
    """FTS5 (3文字以上) と LIKE (2文字以下) を組み合わせたWHERE句を生成"""
    fts_terms = [t for t in tokens if len(t) >= 3]
    like_terms = [t for t in tokens if len(t) < 3]

    clauses: list[str] = []
    params: list[str] = []

    if fts_terms:
        match_query = " AND ".join(fts_terms)  # AND 検索
        clauses.append("docs MATCH ?")
        params.append(match_query)

    for t in like_terms:
        clauses.append("text LIKE ?")
        params.append(f"%{t}%")

    if not clauses:
        # キーワードが空という異常系だが、念の為全件返さないよう FALSE 条件
        clauses.append("0")

    where_sql = " AND ".join(f"({c})" for c in clauses)
    return where_sql, params


def search_database(tokens: list[str], db_path: Path, limit: int = 50, tag_filters: list[str] = []) -> list[tuple]:
    """キーワードリストを受け取り、FTS5+LIKE で検索し、Python 側でタグ絞り込みを行う"""

    results_list: list[tuple] = []

    if not db_path.exists():
        print(f"エラー: データベースファイルが見つかりません: {db_path}", file=sys.stderr)
        return results_list

    where_sql, params = _build_where_clause(tokens)

    try:
        with sqlite3.connect(db_path) as con:
            # docs テーブル確認
            cur = con.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='docs'")
            if cur.fetchone() is None:
                print("エラー: FTS テーブル 'docs' が見つかりません。extract.py を先に実行してください。", file=sys.stderr)
                return results_list

            order_sql = "ORDER BY rank" if any(len(t) >= 3 for t in tokens) else ""

            # LIMIT は十分大きめに取得しておき、Python 側でタグフィルタを適用してから最終件数を制限する
            provisional_limit = max(limit * 5, 500)

            sql = f"""
                SELECT rowid, doc_id, page, paragraph_num, tags,
                       snippet(docs, 0, '>>', '<<', '...', 15) AS snippet_text
                FROM docs
                WHERE {where_sql}
                {order_sql}
                LIMIT ?
            """

            cur = con.execute(sql, (*params, provisional_limit))
            candidates = cur.fetchall()

            if not candidates:
                return results_list

            # --- Python 側でタグフィルタを適用 ---
            if tag_filters:
                normalized_tags = [tg.lower() for tg in tag_filters]
                for rec in candidates:
                    tags_field = (rec[3] or "").lower()
                    if all(tg in tags_field for tg in normalized_tags):
                        results_list.append(rec)
            else:
                results_list = candidates

            # 最終的に LIMIT 件数にトリム
            results_list = results_list[:limit]

    except sqlite3.OperationalError as e:
        print(f"SQLite OperationalError: {e}", file=sys.stderr)
    except Exception as e:
        print(f"検索中に予期せぬエラーが発生しました: {e}", file=sys.stderr)

    return results_list


def format_for_obsidian(query: str, results: list[tuple], tokens: list[str]) -> str:
    """検索結果をObsidian (Markdown) 用に整形する"""
    if not results:
        return "" # 保存する内容がない

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    output_lines = [
        f"# 検索: {query}",
        f"検索日時: {timestamp}",
        f"ヒット件数: {len(results)} 件",
        "---",
        ""
    ]

    for record in results:
        if len(record) == 6:
            rowid, doc_id, page, para_idx, tags_str, snippet_text = record
        elif len(record) == 5:
            # 後方互換: paragraph_num なし
            rowid, doc_id, page, tags_str, snippet_text = record
            para_idx = None
        else:
            rowid, doc_id, page, snippet_text = record
            para_idx = None
            tags_str = ""
        formatted_snippet = " ".join(snippet_text.splitlines()).strip()
        for tk in tokens:
            if len(tk) < 3:
                formatted_snippet = re.sub(re.escape(tk), f'>>{tk}<<', formatted_snippet)
        wrapped_snippet = textwrap.fill(formatted_snippet, width=80, initial_indent='    ', subsequent_indent='    ')
        # Obsidian で PDF を開く際の参考リンク (ファイル名#page=)
        slug = "".join(c if c.isalnum() or c in '_-' else '_' for c in Path(doc_id).stem)[:60]
        note_filename = f"{slug}_p{page:03d}_{para_idx:02d}.md" if para_idx is not None else None
        link_display = f"[[{note_filename}]]" if note_filename else f"{doc_id} p{page}"
        tag_note = f" | Tags: {tags_str}" if tags_str else ""
        para_note = f" | Para: {para_idx}" if para_idx is not None else ""
        output_lines.append(f"## ID: {rowid} | {link_display}{para_note}{tag_note}")
        output_lines.append(f"> {wrapped_snippet}") # snippet を引用ブロックで表示
        output_lines.append("") # 空行

    return "\n".join(output_lines)


def save_to_obsidian(content: str, query: str):
    """整形された内容をObsidianのInbox (または代替パス) に保存する"""
    if not OBSIDIAN_INBOX_PATH:
        print("警告: Obsidian Vault のパス (OBSIDIAN_INBOX_PATH) が config.py に設定されていません。", file=sys.stderr)
        # 代替案: Desktop に保存
        desktop_path = Path.home() / "Desktop"
        target_dir = desktop_path
        print(f"代わりにデスクトップに保存します: {target_dir}", file=sys.stderr)
        # もし Desktop がなければカレントディレクトリに保存
        if not desktop_path.exists():
            target_dir = Path(".")
            print(f"警告: デスクトップが見つかりません。カレントディレクトリに保存します: {target_dir.resolve()}", file=sys.stderr)

    else:
        target_dir = OBSIDIAN_INBOX_PATH
        # 設定されたパスが存在しない場合のフォールバック
        if not target_dir.exists():
             print(f"警告: 設定された Obsidian パスが見つかりません: {target_dir}", file=sys.stderr)
             print(f"代わりにカレントディレクトリに保存します: {Path('.').resolve()}", file=sys.stderr)
             target_dir = Path(".")

    if not content:
        print("情報: 保存する内容がないため、ファイル作成はスキップされました。")
        return

    # ファイル名を生成 (クエリ + 日時)
    # ファイル名に使えない文字を置換 (簡易版)
    safe_query = "".join(c if c.isalnum() else "_" for c in query)
    if len(safe_query) > 50: safe_query = safe_query[:50] + "_etc"
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"search_{safe_query}_{timestamp_str}.md"
    filepath = target_dir / filename

    try:
        # ディレクトリが存在しない場合は作成 (OBSIDIAN_INBOX_PATH の場合のみ)
        if OBSIDIAN_INBOX_PATH and target_dir == OBSIDIAN_INBOX_PATH:
            target_dir.mkdir(parents=True, exist_ok=True)

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"情報: 検索結果を保存しました -> {filepath.resolve()}")

    except OSError as e:
        print(f"エラー: ファイル ({filepath}) の書き込みに失敗しました: {e}", file=sys.stderr)
    except Exception as e:
        print(f"エラー: ファイル保存中に予期せぬ問題が発生しました: {e}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(
        description="SQLite FTS5 テーブルから段落を検索します。",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
        使用例:
          # "AI" と "技術" を含む段落を検索 (デフォルトDB)
          python search.py AI 技術

          # 特定のDBから "マルチカラム" を含む段落を検索し、タグ "layout" で絞り込み
          python search.py マルチカラム --db_path my_docs.sqlite -t layout

          # "セキュリティ" を含む段落を検索し、Obsidian用に保存
          python search.py セキュリティ -o
        """)
    )
    parser.add_argument("keywords", nargs='+', help="検索キーワード (スペース区切り)")
    parser.add_argument("-o", "--obsidian", action="store_true", help="Obsidian 用 Markdown 形式で出力しファイル保存")
    parser.add_argument("-l", "--limit", type=int, default=50, help="最大表示件数")
    parser.add_argument(
        "-t", "--tags", nargs='*', type=str, default=[], help="絞り込みタグ (スペース区切り、AND 検索)"
    )
    parser.add_argument(
        "--db_path",
        type=Path,
        default=DB_PATH,
        help=f"検索対象の SQLite DB ファイル (デフォルト: {DB_PATH})"
    )

    args = parser.parse_args()

    # タグを正規化 (小文字に)
    tag_filters = [_normalize_tag(t) for t in args.tags]

    # 修正: args.db_path を使用
    results = search_database(args.keywords, args.db_path, limit=args.limit, tag_filters=tag_filters)

    if not results:
        print("該当する結果は見つかりませんでした。")
        return

    result_count = len(results)

    if result_count > 0:
        print(f"--- ヒット件数: {result_count} 件 ---")
        for record in results:
            if len(record) == 6:
                rowid, doc_id, page, para_idx, tags_str, snippet_text = record
            elif len(record) == 5:
                # 後方互換: paragraph_num なし
                rowid, doc_id, page, tags_str, snippet_text = record
                para_idx = None
            else:
                rowid, doc_id, page, snippet_text = record
                para_idx = None
                tags_str = ""
            formatted_snippet = " ".join(snippet_text.splitlines()).strip()
            for tk in args.keywords:
                if len(tk) < 3:
                    formatted_snippet = re.sub(re.escape(tk), f'>>{tk}<<', formatted_snippet)
            wrapped_snippet = textwrap.fill(formatted_snippet, width=80, initial_indent='    ', subsequent_indent='    ')
            slug = "".join(c if c.isalnum() or c in '_-' else '_' for c in Path(doc_id).stem)[:60]
            note_filename = f"{slug}_p{page:03d}_{para_idx:02d}.md" if para_idx is not None else None
            link_display = f"[[{note_filename}]]" if note_filename else f"{doc_id} p{page}"
            print(f"ID: {rowid:<5} | {link_display} (p{page}, para {para_idx if para_idx is not None else '-'}) Tags: {tags_str}")
            print(wrapped_snippet)
            print("-" * 10)
        print(f"--- End of Results ---")

    # Obsidian 保存
    if args.obsidian:
        obsidian_content = format_for_obsidian(" ".join(args.keywords), results, args.keywords)
        save_to_obsidian(obsidian_content, " ".join(args.keywords))


# タグ絞り込みを行うための正規化関数
def _normalize_tag(tag: str) -> str:
    """タグ名をDB保存形式 (小文字) に正規化"""
    return tag.strip().lower()


if __name__ == '__main__':
    main() 