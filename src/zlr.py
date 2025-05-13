import argparse
import sys
import os # SCRIPT_DIR のために必要 (Path の前に)
from pathlib import Path # SCRIPT_DIRのために必要

# プロジェクトルートをsys.pathに追加して、各モジュールをインポートしやすくする
# (zlr.pyがルートにある前提)
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from zlr_doctor import run_checks as doctor_run_checks
from extract.__main__ import main as extract_main
from search import main as search_main_func # search.main だと衝突可能性あるので変更
from zlr_watch import main as watch_main

def main():
    parser = argparse.ArgumentParser(
        description="ZLR: PDF Extraction and Recall System",
        formatter_class=argparse.RawTextHelpFormatter,
        prog="zlr" # ヘルプ表示で `zlr.py` の代わりに `zlr` と表示
    )
    subparsers = parser.add_subparsers(dest="command", title="Available commands", help="Run 'zlr <command> -h' for more information on a specific command.")
    subparsers.required = True

    # --- Doctor Command ---
    parser_doctor = subparsers.add_parser(
        "doctor",
        help="Check system dependencies and configuration for ZLR.",
        description="Performs a series of checks to ensure ZLR can run correctly."
    )
    parser_doctor.set_defaults(func=lambda _: doctor_run_checks()) # doctorは引数を取らない

    # --- Extract Command ---
    # extractコマンドは自身の __main__.py で引数をパースするので、ここではヘルプメッセージのみ定義
    parser_extract = subparsers.add_parser(
        "extract",
        help="Extract text from PDF files and save to database. Run 'zlr extract -h' for details.",
        description="Extracts text and metadata from PDF files and stores it in an SQLite FTS5 database.",
        add_help=False # extract側でヘルプを出すので一旦False
    )
    # 実際には extract_main が sys.argv[2:] を見るようにする
    parser_extract.set_defaults(func=lambda _: extract_main()) # extract_main が sys.argv を見る想定

    # --- Search Command ---
    parser_search = subparsers.add_parser(
        "search",
        help="Search the database for paragraphs. Run 'zlr search -h' for details.",
        description="Searches the SQLite database for paragraphs matching keywords and tags.",
        add_help=False
    )
    parser_search.set_defaults(func=lambda _: search_main_func()) # search_main が sys.argv を見る想定

    # --- Watch Command ---
    parser_watch = subparsers.add_parser(
        "watch",
        help="Watch a folder for new PDFs and extract them automatically. Run 'zlr watch -h' for details.",
        description="Monitors a specified folder and automatically extracts new or modified PDF files."
    )
    parser_watch.set_defaults(func=lambda _: watch_main()) # watch_main が sys.argv を見る想定

    # トップレベルの引数とサブコマンドをパース
    # ここで parse_known_args を使い、未知の引数 (各サブコマンドの引数) を補足
    args, unknown_args = parser.parse_known_args()

    # サブコマンドに渡す引数を準備 (サブコマンド名を除いたもの)
    # sys.argv を直接書き換えるのは一般的ではないが、各main関数がsys.argvを直接参照している現在の作りのため、
    # ここでサブコマンド実行前に sys.argv を調整するアプローチを取る。
    # 例: `python zlr.py extract file.pdf --option` の場合、extract_main には `extract file.pdf --option` のように渡したい
    # しかし、各mainは `python <script_name> file.pdf --option` を期待している。
    # よって、sys.argv[0] を `zlr <command>` のようにし、残りを unknown_args にする。
    
    if hasattr(args, 'func'):
        # 各サブコマンドのmain関数が自身のファイル名で始まるsys.argvを期待しているため、
        # sys.argvを一時的に書き換えて実行する。
        original_argv = list(sys.argv)
        if args.command:
            # sys.argv[0] はスクリプト名 (zlr.py) または zlr <command>
            # sys.argv[1:] はサブコマンドの引数群 (unknown_args とほぼ同じはず)
            # 各main関数は自分のファイル名がsys.argv[0]に来ることを期待していないので、
            # むしろコマンド名と引数のみを渡す形に sys.argv を加工する。
            # 例: python search.py keyword -o -> sys.argv = ['search.py', 'keyword', '-o']
            # zlr.py search keyword -o -> search_main()呼び出し時に sys.argv = ['search', 'keyword', '-o'] (この形を目指す)
            # ただし、現状の実装では各スクリプトは自身のファイル名を期待していないため、
            # コマンド名以降を渡せば argparser が機能する。
            sys.argv = [args.command] + unknown_args
            
        try:
            args.func(args) # argsを渡す形に統一。doctorは受け取るが使わない。
        finally:
            sys.argv = original_argv # 必ず元に戻す
    else:
        parser.print_help()

if __name__ == "__main__":
    main() 