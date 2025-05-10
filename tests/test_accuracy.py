# -----------------------------
# テスト: PDF抽出精度一括評価
# -----------------------------
# ・tests/sample_pdfs 内の PDF と tests/ground_truth 内の txt をペアにして類似度を計測
# ・抽出には extract.py の extract_and_process_pdf を使用
# ・結果を JSON に保存し、平均/最小/最大類似度を表示

from __future__ import annotations

# --- 標準ライブラリ ---
import sys
from pathlib import Path
from difflib import SequenceMatcher
import json

# --- サードパーティ ---
import numpy as np

# --- プロジェクト内 ---
# プロジェクトルート (tests/ の 1 つ上) を import パスに追加
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from extract import extract_and_process_pdf  # noqa: E402

def calculate_similarity(text1, text2):
    """2つのテキスト間の類似度を計算"""
    return SequenceMatcher(None, text1, text2).ratio()

def load_ground_truth(pdf_name):
    """PDFに対応する正解テキストを読み込む"""
    ground_truth_dir = Path("tests/ground_truth")
    ground_truth_file = ground_truth_dir / f"{pdf_name}.txt"
    print(f"  [DEBUG] load_ground_truth: pdf_name = '{pdf_name}'")
    print(f"  [DEBUG] load_ground_truth: Attempting to load: {ground_truth_file.resolve()}")
    print(f"  [DEBUG] load_ground_truth: File exists? {ground_truth_file.exists()}")
    
    if not ground_truth_file.exists():
        return None
    return ground_truth_file.read_text(encoding='utf-8')

def extract_text_from_pdf(pdf_path):
    """extract_and_process_pdfの結果からテキスト部分のみを連結して返す"""
    paragraphs = extract_and_process_pdf(Path(pdf_path), patent_mode=False, force_ocr=False)
    if not paragraphs:
        return ""
    # 4番目の要素がテキスト
    texts = [p[3] for p in paragraphs if len(p) > 3]
    return "\n".join(texts)

def main():
    # テスト用PDFディレクトリ
    pdf_dir = Path("tests/sample_pdfs")
    
    # 結果を保存するリスト
    results = []
    
    # 各PDFファイルに対してテストを実行
    for pdf_file in pdf_dir.glob("*.pdf"):
        pdf_name = pdf_file.stem
        print(f"\n処理中: {pdf_name}")
        
        # 正解テキストを読み込む
        ground_truth = load_ground_truth(pdf_name)
        if ground_truth is None:
            print(f"警告: {pdf_name}の正解テキストが見つかりません")
            continue
        
        # PDFからテキストを抽出
        extracted_text = extract_text_from_pdf(str(pdf_file))
        
        # 類似度を計算
        similarity = calculate_similarity(ground_truth, extracted_text)
        
        # 結果を保存
        results.append({
            "pdf_name": pdf_name,
            "similarity": similarity
        })
        
        print(f"類似度: {similarity:.4f}")
    
    # 全体的な結果を表示
    if results:
        similarities = [r["similarity"] for r in results]
        print("\n=== 全体の結果 ===")
        print(f"平均類似度: {np.mean(similarities):.4f}")
        print(f"最小類似度: {np.min(similarities):.4f}")
        print(f"最大類似度: {np.max(similarities):.4f}")
        
        # 結果をJSONファイルに保存
        with open("test_results.json", "w", encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print("\n結果をtest_results.jsonに保存しました")

if __name__ == "__main__":
    main() 