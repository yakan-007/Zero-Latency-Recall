"""tag_utils.py

YAML 形式のタグ辞書を読み込むユーティリティ。

Example
-------
from tag_utils import load_tags
TAGS = load_tags(Path('tags_JP.yaml'))
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import yaml


def load_tags(yaml_path: Path) -> Dict[str, List[str]]:
    """指定された YAML ファイルからタグ辞書を読み込む。

    Parameters
    ----------
    yaml_path : Path
        タグ定義 YAML ファイルのパス。

    Returns
    -------
    dict
        {tag_name: [keywords, ...]} 形式の辞書。
        読み込み失敗時は空辞書を返す。
    """
    if not yaml_path.exists():
        print(f"[tag_utils] タグ辞書ファイルが見つかりません: {yaml_path}")
        return {}

    try:
        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        if not isinstance(data, dict):
            print(f"[tag_utils] YAML 形式が不正です。dict ではありません: {yaml_path}")
            return {}
        # keywords がリストでない場合などを正規化
        normalised: Dict[str, List[str]] = {}
        for tag, body in data.items():
            if isinstance(body, dict):
                kws = body.get("keywords", [])
            else:
                kws = body  # 直接リストが書かれているケース
            if isinstance(kws, str):
                kws = [kws]
            normalised[tag] = list(kws)
        return normalised
    except yaml.YAMLError as e:
        print(f"[tag_utils] YAML 読み込みエラー: {e}")
        return {} 