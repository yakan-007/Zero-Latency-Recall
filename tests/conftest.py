import json
import pytest
from datetime import datetime
from pathlib import Path

# プロジェクトルート (tests/ の 1 つ上)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
JSON_REPORT_PATH = PROJECT_ROOT / "test_results.json"

def pytest_sessionstart(session):
    """テストセッション開始時に呼び出される"""
    session.results = []

@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """各テスト実行後に呼び出される"""
    outcome = yield
    report = outcome.get_result()
    
    if report.when == 'call' and hasattr(item.session, 'results'):
        pdf_name = ""
        similarity = None
        status = "passed" if report.passed else "failed"
        method_info = "" # 将来的に

        for prop, value in item.user_properties:
            if prop == "pdf_name":
                pdf_name = value
            elif prop == "similarity":
                similarity = value
            # elif prop == "method_info": # これはまだ実装されていない
            #     method_info = value
        
        # 類似度テスト以外 (pdf_name が user_properties にないもの) は記録しないか、
        # あるいは別の形で記録する。ここでは pdf_name があるものだけを対象とする。
        if pdf_name: 
            item.session.results.append({
                "timestamp": datetime.now().isoformat(),
                "pdf_name": pdf_name,
                "test_function": item.name,
                "similarity": similarity,
                "status": status,
                "method_info": method_info, 
                "duration": report.duration,
            })

def pytest_sessionfinish(session, exitstatus):
    """テストセッション終了時に呼び出される"""
    if hasattr(session, 'results') and session.results:
        # JSONに書き出すデータは session.results 全体
        results_to_save = session.results
        
        # 統計計算用のリスト (similarity が None でないものから)
        valid_similarities = [r["similarity"] for r in results_to_save if r.get("similarity") is not None]
        
        if valid_similarities:
            avg_sim = sum(valid_similarities) / len(valid_similarities) if len(valid_similarities) > 0 else 0
            min_sim = min(valid_similarities) if len(valid_similarities) > 0 else 0
            max_sim = max(valid_similarities) if len(valid_similarities) > 0 else 0
            print("\n=== 全体の類似度結果 ===")
            print(f"テストケース数 (類似度あり): {len(valid_similarities)}")
            print(f"平均類似度: {avg_sim:.4f}")
            print(f"最小類似度: {min_sim:.4f}")
            print(f"最大類似度: {max_sim:.4f}")
        else:
            print("\n類似度が記録されたテストケースはありませんでした。")

        with open(JSON_REPORT_PATH, "w", encoding="utf-8") as f:
            json.dump(results_to_save, f, ensure_ascii=False, indent=2)
        print(f"\nテスト結果を {JSON_REPORT_PATH} に保存しました。")
    else:
        print(f"\n{JSON_REPORT_PATH} に保存するテスト結果がありません。") 