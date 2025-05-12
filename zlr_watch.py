import time
import logging
import sys # logging設定で使用するため先に import
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import subprocess
from extract.config import WATCH_FOLDER_PATH, DB_PATH

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

class PDFEventHandler(FileSystemEventHandler):
    """PDFファイルのイベントを処理するハンドラ"""
    def __init__(self, debounce_time=5.0):
        super().__init__()
        self.last_event_time = {}
        self.debounce_time = debounce_time # 秒

    def _should_process(self, event_path_str: str) -> bool:
        """イベントを処理すべきか (デバウンス制御)"""
        current_time = time.time()
        last_time = self.last_event_time.get(event_path_str, 0)
        if current_time - last_time < self.debounce_time:
            logging.debug(f"Debounced event for: {event_path_str}")
            return False
        self.last_event_time[event_path_str] = current_time
        return True

    def _run_extraction(self, event_path_str: str):
        """指定されたPDFファイルに対して extract.py を実行する"""
        try:
            pdf_path = Path(event_path_str)
            if pdf_path.exists() and pdf_path.suffix.lower() == ".pdf":
                cmd = ["python", "-m", "extract", str(pdf_path)]
                logging.info(f"↓↓↓ extract モジュールを実行します: {' '.join(cmd)} ↓↓↓")
                # extract.py に --patent オプションが必要な場合のロジックは別途検討 -> 不要に (edition オプションはあるが、watchでは通常抽出)
                # ここではシンプルに通常PDFとして処理
                result = subprocess.run(cmd, capture_output=True, text=True, check=False)
                if result.returncode == 0:
                    logging.info(f"extract.py 正常終了: {event_path_str}")
                    logging.debug(f"extract.py stdout:\n{result.stdout}")
                else:
                    logging.error(f"extract.py がエラー終了しました (コード: {result.returncode}): {event_path_str}")
                    logging.error(f"extract.py stderr:\n{result.stderr}")
                logging.info(f"↑↑↑ extract.py の実行完了: {event_path_str} ↑↑↑")

        except FileNotFoundError:
            logging.error(f"extract.py が見つかりません。パスを確認してください。")
        except Exception as e:
            logging.error(f"Error running extraction for {event_path_str}: {e}", exc_info=True)

    def on_created(self, event):
        if not event.is_directory and event.src_path.lower().endswith(".pdf"):
            if self._should_process(event.src_path):
                logging.info(f"New PDF detected: {event.src_path}")
                self._run_extraction(event.src_path)

    def on_modified(self, event):
        if not event.is_directory and event.src_path.lower().endswith(".pdf"):
            if self._should_process(event.src_path):
                logging.info(f"PDF modified: {event.src_path}")
                # TODO: 変更時はDBから既存データを削除する前処理が必要
                # 現状は新規作成と同様に処理 (上書きはされるが古いデータが残る可能性)
                logging.warning(f"Handling modification for {event.src_path} as new creation. Consider pre-deletion from DB.")
                self._run_extraction(event.src_path)

    def on_moved(self, event):
        if not event.is_directory and event.dest_path.lower().endswith(".pdf"):
            if Path(event.dest_path).parent == WATCH_FOLDER_PATH:
                 if self._should_process(event.dest_path):
                    logging.info(f"PDF moved into watched folder: {event.dest_path}")
                    self._run_extraction(event.dest_path)
            elif Path(event.src_path).parent == WATCH_FOLDER_PATH:
                 logging.info(f"PDF moved out of watched folder: {event.src_path}")
                 # TODO: DBから該当ファイルを削除する処理
                 logging.warning(f"PDF {event.src_path} moved out. Manual DB cleanup might be needed.")


def main():
    if not WATCH_FOLDER_PATH or not WATCH_FOLDER_PATH.exists() or not WATCH_FOLDER_PATH.is_dir():
        logging.error(f"監視対象フォルダが無効です: {WATCH_FOLDER_PATH}")
        return

    logging.info(f"Watching folder: {WATCH_FOLDER_PATH} for new/modified PDF files.")
    event_handler = PDFEventHandler()
    observer = Observer()
    observer.schedule(event_handler, str(WATCH_FOLDER_PATH), recursive=False) 
    observer.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
        logging.info("Watcher stopped by user.")
    except Exception as e:
        observer.stop()
        logging.error(f"An unexpected error occurred: {e}", exc_info=True)
    observer.join()

if __name__ == "__main__":
    main() 