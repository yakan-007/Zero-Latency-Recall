#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
ロギングシステムのデバッグ用スクリプト
"""

import sys
import os
import logging
import pdfplumber

# 実行環境情報の表示
print("===== 実行環境情報 =====")
print(f"Python バージョン: {sys.version}")
print(f"実行パス: {sys.executable}")
print(f"現在の作業ディレクトリ: {os.getcwd()}")
print(f"sys.path: {sys.path}")
print("=======================\n")

# 標準出力の確認
print("標準出力テスト: これが表示されますか？")
sys.stdout.flush()  # バッファをフラッシュ

# 標準エラー出力の確認
print("標準エラー出力テスト", file=sys.stderr)
sys.stderr.flush()  # バッファをフラッシュ

# ロギングの初期化（ファイル出力も追加）
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("debug.log"),  # ファイルにも出力
        logging.StreamHandler(sys.stdout)  # 標準出力にも出力
    ]
)

# ルートロガーのレベル確認
root_logger = logging.getLogger()
print(f"ルートロガーのレベル: {logging.getLevelName(root_logger.level)}")

# 各モジュールのロガーレベルを確認・設定
loggers = [
    logging.getLogger("pdfplumber"),
    logging.getLogger("PIL"),
    logging.getLogger("pdf2image"),
    logging.getLogger()  # ルートロガー
]

print("\n===== 現在のロガー設定 =====")
for logger in loggers:
    print(f"Logger: {logger.name}, Level: {logging.getLevelName(logger.level)}")
print("==========================\n")

# pdfplumberのロガーレベルを明示的に設定（問題の原因かもしれない）
pdfplumber_logger = logging.getLogger("pdfplumber")
pdfplumber_logger.setLevel(logging.WARNING)  # WARNINGレベル以上のみ表示

# テストログの出力
logging.debug("これはデバッグメッセージです")
logging.info("これは情報メッセージです")
logging.warning("これは警告メッセージです")
logging.error("これはエラーメッセージです")

# pdfplumberの直接インポート確認
print("\n===== pdfplumberモジュール情報 =====")
print(f"pdfplumber のバージョン: {pdfplumber.__version__ if hasattr(pdfplumber, '__version__') else '不明'}")
print(f"pdfplumber のパス: {pdfplumber.__file__}")
print("===============================\n")

print("スクリプト実行完了！ debug.log ファイルも確認してください。") 