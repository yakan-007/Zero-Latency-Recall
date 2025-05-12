# config.py
"""共通設定モジュール
.env から PDF_PATH, DB_PATH, OBSIDIAN_VAULT を読み込み、Path 型で公開します。
他モジュールはこれを import して参照してください。"""

import os
from pathlib import Path
from dotenv import load_dotenv

# .env 読み込み
# print("DEBUG: Attempting to load .env file...") # デバッグ削除
env_path = Path('.') / '.env' # カレントディレクトリの .env を明示
# print(f"DEBUG: Explicit .env path: {env_path.resolve()}") # デバッグ削除
load_success = load_dotenv(dotenv_path=env_path, override=True) # 明示的にパスを指定し、既存の環境変数を上書き
# print(f"DEBUG: load_dotenv() executed. Success: {load_success}") # デバッグ削除

# 環境変数を直接取得して表示 (デバッグ用だったので削除)
# pdf_path_env = os.getenv("PDF_PATH")
# db_path_env = os.getenv("DB_PATH")
# obsidian_vault_env = os.getenv("OBSIDIAN_VAULT")
# print(f"DEBUG: Raw environment variables:")
# print(f"DEBUG:   PDF_PATH={pdf_path_env}")
# print(f"DEBUG:   DB_PATH={db_path_env}")
# print(f"DEBUG:   OBSIDIAN_VAULT={obsidian_vault_env}")

# デフォルト値を持ちつつ環境変数で上書き
PDF_PATH: Path = Path(os.getenv("PDF_PATH", "sample.pdf")) # 元の取得方法に戻す (load_dotenvで設定されるため)
DB_PATH: Path = Path(__file__).parent / "zlr.sqlite"
# print(f"DEBUG: Final PDF_PATH in config.py: {PDF_PATH}") # デバッグ削除

# Obsidian Vault 関連の設定
OBSIDIAN_VAULT_BASE_PATH_STR = os.getenv("OBSIDIAN_VAULT") # 元の取得方法に戻す
if not OBSIDIAN_VAULT_BASE_PATH_STR:
    # 環境変数がない場合は、ファイル保存機能を無効化するか、エラーにするか選択
    # ここでは警告を出し、ファイル保存パスをNoneにして機能が動作しないようにする
    print("警告: 環境変数 OBSIDIAN_VAULT が設定されていません。Obsidianへの自動保存は無効になります。")
    OBSIDIAN_INBOX_PATH: Path | None = None
else:
    OBSIDIAN_VAULT_BASE_PATH = Path(OBSIDIAN_VAULT_BASE_PATH_STR)
    OBSIDIAN_INBOX_PATH: Path | None = OBSIDIAN_VAULT_BASE_PATH / "zlr-inbox"

WATCH_FOLDER_PATH = Path("/Users/leo/leona_ai_madeproject/zlr-dev/sample/regular")

__all__ = ["PDF_PATH", "DB_PATH", "OBSIDIAN_INBOX_PATH"] 