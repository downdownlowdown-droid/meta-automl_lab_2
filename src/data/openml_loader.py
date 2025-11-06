from __future__ import annotations
import os
import random
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, List
import yaml

# Настроим пользовательский каталог для OpenML ДО импорта openml
_OPENML_DIR = "/content/.openml"
os.makedirs(_OPENML_DIR, exist_ok=True)
os.environ["OPENML_CONFIG_DIR"] = _OPENML_DIR
os.environ["OPENML_CACHE_DIR"] = _OPENML_DIR

import openml  # noqa: E402

from src.utils.logging_utils import timeit

# На всякий случай продублируем конфиг через API либы
openml.config.set_root_cache_directory(_OPENML_DIR)

RANDOM = 42

def load_config(cfg_path: str) -> Dict[str, Any]:
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def list_openml_datasets(cfg: Dict[str, Any]) -> pd.DataFrame:
    """Запрашиваем список датасетов у OpenML и фильтруем по метаданным."""
    o = cfg["openml"]
    with timeit("OpenML: list_datasets"):
        df = openml.datasets.list_datasets(output_format="dataframe")
    # Базовые фильтры по метаданным
    df = df[df["NumberOfInstances"].between(o["min_rows"], o["max_rows"])]
    df = df[df["NumberOfFeatures"] <= o["max_features"]]
    # Классификация: минимум 2 класса
    df = df[df["NumberOfClasses"].fillna(0) >= 2]
    # Исключим заведомо "мертвые"/неактивные
    if "status" in df.columns:
        df = df[df["status"].fillna("active").str.lower().eq("active")]
    # Огрублённая оценка пропусков по метаданным (на этапе загрузки проверим точнее)
    if "NumberOfMissingValues" in df.columns:
        miss = df["NumberOfMissingValues"].fillna(0)
        rows = df["NumberOfInstances"].replace(0, np.nan)
        approx_miss_ratio = miss / (rows * df["NumberOfFeatures"].clip(lower=1))
        max_miss = o.get("max_missing_ratio", None) # Added check for max_missing_ratio
        if max_miss is not None:
            df = df[approx_miss_ratio.fillna(0) <= float(max_miss)]
    # Чистим экзотические датасеты без названия/целевой
    df = df[~df["name"].isna()]
    # Набросим случайный порядок для разнообразия
    rnd = np.random.default_rng(cfg["selection"]["random_state"])
    df = df.sample(frac=1.0, random_state=cfg["selection"]["random_state"])
    # Возьмём верхние desired_count
    top_n = int(cfg["selection"]["desired_count"])
    df = df.head(top_n * 3)  # возьмём с запасом, часть отсеется потом на точной проверке
    return df.reset_index().rename(columns={"did": "openml_id"})

def refine_on_load(df_ids: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame:
    """
    Делаем более строгую проверку: пробуем загрузить описание датасета,
    выясняем целевую, типы признаков и реальную долю пропусков в X/target.
    Ничего не качаем целиком — только метаинфу, чтобы не тратить трафик.
    """
    rows = []
    keep = []
    with timeit("OpenML: refine and build dataset cards"):
        for _, row in df_ids.iterrows():
            did = int(row["openml_id"])
            try:
                ds = openml.datasets.get_dataset(did, download_data=False)
                # Иногда у датасета нет адекватного target — пропускаем
                target = (ds.default_target_attribute or "").strip()
                if not target or target in cfg["openml"].get("target_blacklist", []):
                    continue
                qualities = ds.qualities or {}
                nrows = int(qualities.get("NumberOfInstances", row.get("NumberOfInstances", np.nan)) or 0)
                nfeat = int(qualities.get("NumberOfFeatures", row.get("NumberOfFeatures", np.nan)) or 0)
                nclasses = int(qualities.get("NumberOfClasses", row.get("NumberOfClasses", 0)) or 0)
                # Точная оценка пропусков, если есть качество MissingValues
                miss_cells = qualities.get("NumberOfMissingValues")
                approx_miss_ratio = None
                if miss_cells is not None and nrows and nfeat:
                    approx_miss_ratio = float(miss_cells) / (nrows * nfeat)
                # Сжимаем карточку
                rows.append({
                    "openml_id": did,
                    "name": ds.name,
                    "target": target,
                    "n_rows": nrows,
                    "n_features": nfeat,
                    "n_classes": nclasses,
                    "approx_missing_ratio": approx_miss_ratio,
                    "status": row.get("status", "active"),
                })
            except Exception:
                # Любые странные наборы — мимо
                continue
    cards = pd.DataFrame(rows)
    if cards.empty:
        return cards
    # Итоговые фильтры, уже строже:
    o = cfg["openml"]
    cards = cards[cards["n_rows"].between(o["min_rows"], o["max_rows"])]
    cards = cards[cards["n_features"] <= o["max_features"]]
    cards = cards[cards["n_classes"] >= 2]
    if o.get("max_missing_ratio") is not None:
        mr = cards["approx_missing_ratio"].fillna(0.0)
        cards = cards[mr <= float(o["max_missing_ratio"])]
    # Дедупликация: одно имя — один датасет (берём меньший по n_rows)
    cards = cards.sort_values(["name", "n_rows"]).drop_duplicates("name", keep="first")
    # Финальная нарезка до нужного количества
    desired = int(cfg["selection"]["desired_count"])
    cards = cards.head(desired).reset_index(drop=True)
    return cards

def build_and_save_list(cfg_path: str):
    cfg = load_config(cfg_path)
    meta_dir = Path(cfg["paths"]["meta_dir"])
    meta_dir.mkdir(parents=True, exist_ok=True)
    # 1) список по метаданным
    df_meta = list_openml_datasets(cfg)
    # 2) уточнение и карточки
    cards = refine_on_load(df_meta, cfg)
    # 3) сохранить
    out_csv = Path(cfg["paths"]["list_csv"])
    cards.to_csv(out_csv, index=False)
    print(f"Saved: {out_csv.resolve()}  ({len(cards)} datasets)")
    return cards

if __name__ == "__main__":
    build_and_save_list("configs/run.yaml")
