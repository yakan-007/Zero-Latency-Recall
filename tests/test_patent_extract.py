import pytest
from pathlib import Path
import sys

# プロジェクトルートをsys.pathに追加 (extractモジュールをインポートするため)
# このテストファイルが tests/ にあることを想定
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from extract import extract_and_process_pdf, TAG_DICT # TAG_DICTもインポート

# テスト対象のPDFファイルリスト
TEST_PATENT_PDFS = [
    project_root / "samples" / "patents" / "JPA 2021091459-000000.pdf",
    project_root / "samples" / "patents" / "JPA 2025509103-000000.pdf",
    project_root / "samples" / "patents" / "JPU 003251040-000000.pdf",
]

# 存在しないPDFを除外
EXISTING_TEST_PATENT_PDFS = [p for p in TEST_PATENT_PDFS if p.exists()]

@pytest.mark.parametrize("pdf_path", EXISTING_TEST_PATENT_PDFS)
def test_patent_extract_finds_claims(pdf_path):
    """特許モードで各PDFから抽出し、「請求項」テキストと 'claims' タグの存在をテスト"""
    print(f"\nTesting PDF: {pdf_path.name}") # どのファイルをテスト中か出力
    # extract_and_process_pdf は (filename, page_num, para_idx, text, source, tags) のリストを返す
    extracted_data = extract_and_process_pdf(pdf_path, patent_mode=True)

    assert extracted_data, f"{pdf_path.name}: 抽出されたデータがありません"

    found_claim_text = False
    found_claim_tag = False

    # 抽出された段落のテキスト内容とタグをチェック
    for _filename, _page_num, _para_idx, content, _source, tags in extracted_data:
        # デバッグ用にいくつか段落を出力
        # print(f"  - Page {_page_num}-{_para_idx} ({_source}) Tags: {tags} | {content[:50]}...")
        if "請求項" in content: # テキスト内容のチェック
            found_claim_text = True
            print(f"  Found '請求項' in: {content[:50]}...") # 見つかった箇所をログ出力
        if "claims" in tags: # タグ付けのチェック
            found_claim_tag = True
            print(f"  Found 'claims' tag in: {content[:50]}...") # 見つかった箇所をログ出力

    assert found_claim_text, f"{pdf_path.name}: 抽出された段落に「請求項」の文字列が見つかりませんでした。TAG_DICT: {TAG_DICT.get('claims')}"
    assert found_claim_tag, f"{pdf_path.name}: 抽出された段落に 'claims' タグが付与されていませんでした。TAG_DICT: {TAG_DICT.get('claims')}"

# 他にも、背景技術や発明の効果など、他のタグについてもテストケースを追加できます。
# 例:
# @pytest.mark.parametrize("pdf_path", EXISTING_TEST_PATENT_PDFS)
# def test_patent_extract_finds_background(pdf_path):
#     extracted_data = extract_and_process_pdf(pdf_path, patent_mode=True)
#     assert any("background" in tags for _, _, _, _, _, tags in extracted_data), \
#         f"{pdf_path.name}: 抽出された段落に 'background' タグが付与されていませんでした。TAG_DICT: {TAG_DICT.get('background')}" 