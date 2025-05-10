import sys
from pathlib import Path

# プロジェクトルートを import パスに追加
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from extract import extract_and_process_pdf

def main():
    pdf_name = "02_multi-column_2.pdf"
    pdf_path = Path(f"tests/sample_pdfs/{pdf_name}")

    if not pdf_path.exists():
        print(f"エラー: PDFファイルが見つかりません: {pdf_path}")
        return

    print(f"--- {pdf_name} から抽出されたテキスト ---")
    
    # patent_mode や force_ocr の設定は、現在の test_accuracy.py での呼び出し方に合わせる
    # 現在の test_accuracy.py の extract_text_from_pdf -> extract_and_process_pdf の呼び出しでは
    # patent_mode=False, force_ocr=True が設定されている
    extracted_paragraphs = extract_and_process_pdf(pdf_path, patent_mode=False, force_ocr=True)

    if not extracted_paragraphs:
        print("テキストは抽出されませんでした。")
        return

    for i, para_data in enumerate(extracted_paragraphs):
        # para_data: (filename, page_num, para_idx, text, source, tags)
        page_num = para_data[1]
        para_idx_in_page = para_data[2]
        text = para_data[3]
        source = para_data[4]
        
        print(f"\n--- ページ: {page_num}, 段落(ページ内): {para_idx_in_page + 1}, ソース: {source} ---")
        print(text)
        print("---")

if __name__ == "__main__":
    main() 