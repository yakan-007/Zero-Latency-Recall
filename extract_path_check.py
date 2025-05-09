import os
import sys
import inspect

print("\n\n================== PATH CHECK ==================")
print(f"現在実行されているスクリプト: {os.path.abspath(__file__)}")
print(f"現在の作業ディレクトリ: {os.getcwd()}")
print(f"sys.path (Pythonのモジュール検索パス):")
for p in sys.path:
    print(f"  - {p}")
print(f"sys.executable (Python実行ファイル): {sys.executable}")
print("================================================\n\n") 