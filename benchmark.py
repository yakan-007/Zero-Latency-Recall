# benchmark.py
"""PDF処理と検索のベンチマークを測定するスクリプト"""

import time
import sqlite3
import numpy as np
from pathlib import Path
import sys
import os # osモジュールをインポート

# プロジェクトルートをPythonパスに追加
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from config import PDF_PATH, DB_PATH
# extract.py の主要な関数を直接呼び出す代わりに、
# extract.py の main 処理を模倣するか、関数をインポートする
from extract import init_db as extract_init_db, extract_paragraphs
from search import search as zlr_search # search.search と区別するためエイリアス

DEFAULT_NUM_TRIALS = 10 # 各キーワードでのデフォルト試行回数

def benchmark_extraction(pdf_file_path: Path):
    """PDFからのデータ抽出とDB構築の時間を計測"""
    print(f"\n--- 抽出処理のベンチマーク ({pdf_file_path.name}) ---")
    start_time = time.perf_counter()

    # extract.py の main() 関数の主要処理を再現
    extract_init_db() # DBを初期化
    paragraphs = extract_paragraphs(pdf_file_path) # PDFから段落抽出
    if paragraphs:
        with sqlite3.connect(DB_PATH) as conn:
            conn.executemany("INSERT INTO paragraphs VALUES (?,?,?)", paragraphs)
        print(f"{len(paragraphs)} レコードを {DB_PATH} に保存しました。")
    else:
        print("抽出された段落はありませんでした。")

    end_time = time.perf_counter()
    duration = end_time - start_time
    print(f"抽出とDB構築の所要時間: {duration:.4f} 秒")
    return duration

def benchmark_search(keywords_list: list[list[str]], num_trials: int = DEFAULT_NUM_TRIALS):
    """指定されたキーワードリストで検索時間を複数回計測し、統計情報を表示"""
    print(f"\n--- 検索処理のベンチマーク (各{num_trials}回実行) ---")
    all_results = {}

    for keywords in keywords_list:
        search_times = []
        print(f"\n検索キーワード: {keywords}")
        for i in range(num_trials):
            # search関数はコンソール出力とファイル保存も行うため、純粋な検索速度とは言えないが
            # 実用的な応答時間として計測する
            # より厳密にはsearch関数を改修し、検索コアロジックのみを呼び出す
            start_time = time.perf_counter()
            zlr_search(keywords) # search.py の search 関数を呼び出し
            end_time = time.perf_counter()
            search_times.append(end_time - start_time)
            # 検索結果の表示を抑制したい場合は、search.py側で制御するか、
            # ここで出力をキャプチャして捨てるなどの工夫が必要
            if i == 0: print("(検索結果の表示は最初の1回のみ行います。残りは時間計測のみ)")
            if i > 0: # 2回目以降はsearch.pyの出力を少し抑制（完璧ではない）
                sys.stdout = open(os.devnull, 'w') # 標準出力を抑制
                zlr_search(keywords)
                sys.stdout = sys.__stdout__ # 標準出力を元に戻す

        if not search_times:
            print("計測データがありません。")
            continue

        avg_time = np.mean(search_times)
        p50_time = np.percentile(search_times, 50)
        p90_time = np.percentile(search_times, 90)
        p95_time = np.percentile(search_times, 95)
        p99_time = np.percentile(search_times, 99)
        max_time = np.max(search_times)
        min_time = np.min(search_times)

        print(f"  試行回数: {len(search_times)}")
        print(f"  平均時間: {avg_time:.4f} 秒")
        print(f"  最小時間: {min_time:.4f} 秒")
        print(f"  最大時間: {max_time:.4f} 秒")
        print(f"  P50 (中央値): {p50_time:.4f} 秒")
        print(f"  P90: {p90_time:.4f} 秒")
        print(f"  P95: {p95_time:.4f} 秒")
        print(f"  P99: {p99_time:.4f} 秒")
        all_results[tuple(keywords)] = {
            "avg": avg_time, "p95": p95_time, "times": search_times
        }
    return all_results

if __name__ == "__main__":
    # ベンチマーク対象のPDFを指定
    # target_pdf_name = "sample.pdf" # 既存のPDFパス
    # target_pdf_name = "N2959KI.pdf" # 古いテストPDF
    target_pdf_name = "20250501g000980002.pdf" # 新しい官報PDF
    current_pdf_path = project_root / target_pdf_name

    if not current_pdf_path.exists():
        print(f"エラー: ベンチマーク用PDFが見つかりません: {current_pdf_path}")
        sys.exit(1)

    # 1. 抽出処理のベンチマーク
    benchmark_extraction(current_pdf_path) # 指定したPDFで実行

    # 2. 検索処理のベンチマーク
    #    キーワードは新しいPDFの内容に合わせて調整が必要な場合がある
    search_keywords_to_test = [
        ["物語"], # 小説なのでありそうな単語
        ["登場人物"], # 小説なのでありそうな単語
        ["場面転換"],
        ["昔々"], # 古典的な始まり
        ["そして"], # 接続詞
    ]
    # 試行回数を少なく設定（CIなどでは増やす）
    benchmark_search(search_keywords_to_test, num_trials=5)

    print("\nベンチマーク完了。")
    # 将来的には、大きなPDFファイルで再度実行し、P95 <= 0.2s の目標と比較する。 