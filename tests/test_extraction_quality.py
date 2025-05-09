import pytest
import sqlite3
from pathlib import Path
import subprocess
import sys

# プロジェクトルートのパスを取得 (zlr-dev)
PROJECT_ROOT = Path(__file__).parent.parent 

# テスト用PDFがあるディレクトリ
SAMPLE_PDF_DIR = Path(__file__).parent / "sample_pdfs"
# テスト用DBのパス (プロジェクトルート直下に作成)
TEST_DB_PATH = PROJECT_ROOT / "test_zlr.sqlite" 
# extract.py のパス
EXTRACT_PY_PATH = PROJECT_ROOT / "extract.py"


def run_extract_for_test(pdf_path: Path, db_path: Path, patent_mode: bool = False):
    """テスト用に extract.py を実行するヘルパー関数"""
    if db_path.exists():
        db_path.unlink() # 既存のテストDBを削除
    
    cmd = [sys.executable, str(EXTRACT_PY_PATH), str(pdf_path), "--db_path", str(db_path)]
    if patent_mode:
        cmd.append("--patent")
        
    print(f"Running command: {' '.join(cmd)}") # デバッグ用にコマンドを表示
    result = subprocess.run(cmd, capture_output=True, text=True, check=False, encoding='utf-8')
    
    # extract.py の出力を常に表示する
    print(f"extract.py stdout:\\n{result.stdout}")
    print(f"extract.py stderr:\\n{result.stderr}")
    
    assert result.returncode == 0, f"extract.py failed for {pdf_path} with code {result.returncode}"

def get_paragraphs_from_db(db_path: Path, doc_id_like: str) -> list[dict]:
    """テスト用DBから指定されたdoc_idの段落を取得するヘルパー関数"""
    results = []
    print(f"[get_paragraphs_from_db] Attempting to read DB: {db_path}")
    if not db_path.exists():
        print(f"[get_paragraphs_from_db] DB file not found: {db_path}")
        return results
    print(f"[get_paragraphs_from_db] DB file exists. Size: {db_path.stat().st_size} bytes")
        
    with sqlite3.connect(db_path) as con:
        con.row_factory = sqlite3.Row
        # docs テーブルが存在するか確認
        cur_check = con.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='docs';")
        table_exists = cur_check.fetchone()
        if table_exists is None:
            print(f"[get_paragraphs_from_db] Table 'docs' not found in {db_path}")
            return results
        print(f"[get_paragraphs_from_db] Table 'docs' found.")
            
        # まずは条件なしで全件取得を試みる (デバッグ用)
        all_rows_cur = con.execute("SELECT doc_id, text, page, tags FROM docs")
        all_rows = all_rows_cur.fetchall()
        print(f"[get_paragraphs_from_db] Total rows in 'docs' table (unfiltered): {len(all_rows)}")
        if all_rows:
            print(f"[get_paragraphs_from_db] First few doc_ids (unfiltered): {[row['doc_id'] for row in all_rows[:5]]}")

        # 元のクエリ (LIKE は FTS5 UNINDEXED 列ではヒットしない場合がある)
        query = "SELECT text, page, tags FROM docs WHERE doc_id LIKE ?"
        params = (f"%{doc_id_like}%",)
        print(f"[get_paragraphs_from_db] Executing query: {query} with params: {params}")
        cur = con.execute(query, params)
        rows = cur.fetchall()
        print(f"[get_paragraphs_from_db] Rows found with LIKE: {len(rows)}")

        # LIKE でヒットしなかった場合、完全一致でも検索してみる
        if len(rows) == 0:
            print("[get_paragraphs_from_db] LIKE が 0 件のため、完全一致 (=) で再検索します。")
            equality_query = "SELECT text, page, tags FROM docs WHERE doc_id = ?"
            equality_params = (doc_id_like,)
            cur = con.execute(equality_query, equality_params)
            rows = cur.fetchall()
            print(f"[get_paragraphs_from_db] Rows found with equality: {len(rows)}")

        for row in rows:
            results.append(dict(row))
        print(f"[get_paragraphs_from_db] Rows found with doc_id_like '{doc_id_like}': {len(results)}")
    return results

# --- Test Cases ---

def test_placeholder_extraction():
    """
    tests/sample_pdfs ディレクトリ内の最初のPDFファイルに対して、
    extract.py がエラーなく実行できることを確認する基本的なテスト。
    具体的な内容は今後追加していく。
    """
    
    pdf_files = list(SAMPLE_PDF_DIR.glob("*.pdf"))
    assert pdf_files, "テスト用のPDFファイルが tests/sample_pdfs に見つかりません。"
    
    target_pdf = pdf_files[0] # 最初のPDFファイルを選択
    
    print(f"Testing with PDF: {target_pdf.name}")
    run_extract_for_test(target_pdf, TEST_DB_PATH)
    
    # ここでは extract.py が正常終了したことのみを確認
    # 今後、DBの内容を検証するアサーションを追加
    assert TEST_DB_PATH.exists(), "テスト用データベースが作成されていません。"

    # 簡単なDB内容チェック (例: docsテーブルから何か取得できるか)
    paragraphs = get_paragraphs_from_db(TEST_DB_PATH, target_pdf.stem) # PDFのファイル名(拡張子なし)で検索
    assert len(paragraphs) > 0, f"{target_pdf.name} から段落が抽出され、DBに保存されていません。"
    
    print(f"Successfully extracted {len(paragraphs)} paragraphs from {target_pdf.name}") 