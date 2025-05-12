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

from extract.core.extractor import AdvancedExtractor, SimpleExtractor, extract_and_process_pdf

# extract.py にある正規化ユーティリティを再利用
import importlib
_norm = getattr(importlib.import_module("extract.core.extractor"), "_normalize_text_for_compare")

def calculate_similarity(text1, text2):
    """2つのテキスト間の類似度を計算"""
    return SequenceMatcher(None, _norm(text1), _norm(text2)).ratio()

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

def extract_text_from_pdf(pdf_path: Path, extractor_type: str = "advanced"):
    """PDFからテキストを抽出する"""
    paragraphs = []
    if extractor_type == "advanced":
        # AdvancedExtractor を使用
        extractor = AdvancedExtractor(patent_mode=False)
        paragraphs = extractor.process_pdf(pdf_path)
    elif extractor_type == "simple":
        extractor = SimpleExtractor(patent_mode=False)
        paragraphs = extractor.process_pdf(pdf_path)
    else:
        # デフォルトは extract.py の関数 (互換性のため残す場合)
        # またはエラーとする
        paragraphs = extract_and_process_pdf(pdf_path, patent_mode=False, force_ocr=False)

    if not paragraphs:
        return ""
    # 4番目の要素がテキスト
    texts = [p[3] for p in paragraphs if len(p) > 3]
    return "\n".join(texts)

def main():
    # テスト用PDFディレクトリ
    pdf_dir = Path("tests/sample_pdfs")
    
    extractor_types_to_test = ["simple"]

    for current_extractor_type in extractor_types_to_test:
        print(f"\n{'='*10} Testing Extractor: {current_extractor_type.upper()} {'='*10}")
    results = []
    # 各PDFファイルに対してテストを実行
    for pdf_file in pdf_dir.glob("*.pdf"):
        pdf_name = pdf_file.stem
        print(f"\n処理中 ({current_extractor_type}): {pdf_name}")

        # 正解テキストを読み込む
        ground_truth = load_ground_truth(pdf_name)
        if ground_truth is None:
            print(f"警告: {pdf_name}の正解テキストが見つかりません")
            continue
        
        # PDFからテキストを抽出
        extracted_text = extract_text_from_pdf(pdf_file, extractor_type=current_extractor_type)
        
        # 類似度を計算
        similarity = calculate_similarity(ground_truth, extracted_text)
        
        # 結果を保存
        results.append({
            "pdf_name": pdf_name,
            "similarity": similarity,
            "extractor": current_extractor_type
        })
        
        print(f"類似度 ({current_extractor_type}): {similarity:.4f}")
    
    # 全体的な結果を表示
    if results:
        similarities = [r["similarity"] for r in results]
        print(f"\n=== 全体の結果 ({current_extractor_type.upper()}) ===")
        print(f"平均類似度: {np.mean(similarities):.4f}")
        print(f"最小類似度: {np.min(similarities):.4f}")
        print(f"最大類似度: {np.max(similarities):.4f}")
        
        # 結果をJSONファイルに保存
        output_filename = f"test_results_{current_extractor_type}.json"
        with open(output_filename, "w", encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\n結果を{output_filename}に保存しました")

if __name__ == "__main__":
    main() 