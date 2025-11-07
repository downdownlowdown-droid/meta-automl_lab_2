from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional, Union
from collections import Counter

# --- утилиты ---------------------------------------------------------------

def _entropy_from_counts(counts: np.ndarray) -> float:
    p = counts / counts.sum()
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum()) if p.size else 0.0

def _is_categorical(series: pd.Series, max_unique_for_int: int = 20) -> bool:
    # cat dtype или object → категориальные
    if pd.api.types.is_categorical_dtype(series) or pd.api.types.is_object_dtype(series):
        return True
    # целочисленные низкой кардинальности считаем категориальными
    if pd.api.types.is_integer_dtype(series):
        nun = series.nunique(dropna=True)
        return nun <= max_unique_for_int
    return False

def _safe_numeric(s: pd.Series) -> Optional[pd.Series]:
    if pd.api.types.is_numeric_dtype(s):
        return s
    try:
        return pd.to_numeric(s, errors="coerce")
    except Exception:
        return None

# --- основной API ----------------------------------------------------------

@dataclass
class MetaFeatures:
    # базовые
    n_rows: int
    n_features: int
    n_classes: int
    n_cat_features: int
    missing_ratio: float
    class_entropy: float
    # статистические
    mean_feature_mean: float
    mean_feature_std: float
    mean_skewness: float
    mean_kurtosis: float
    avg_abs_corr_num: float
    share_zero_var: float
    # структурные
    share_categorical: float
    share_constant_features: float
    avg_unique_values_per_feature: float
    ratio_num_to_cat: float
    avg_cat_cardinality: float
    class_gini: float

    def to_dict(self) -> Dict[str, Union[int, float]]:
        return self.__dict__.copy()

def compute_meta_features(
    X: pd.DataFrame,
    y: Union[pd.Series, np.ndarray, list],
    *,
    max_unique_for_int: int = 20
) -> MetaFeatures:
    X = X.copy()
    y = pd.Series(y).copy()

    n_rows = int(X.shape[0])
    n_features = int(X.shape[1])

    # типы признаков
    is_cat = [ _is_categorical(X[col], max_unique_for_int=max_unique_for_int) for col in X.columns ]
    n_cat_features = int(np.sum(is_cat))
    n_num_features = n_features - n_cat_features

    # пропуски
    total = n_rows * max(1, n_features)
    missing_ratio = float(X.isna().sum().sum() / total) if total > 0 else 0.0

    # классы
    y_clean = pd.Series(y).dropna()
    class_counts = y_clean.value_counts().values.astype(float)
    n_classes = int(len(class_counts)) if len(class_counts) else 0
    class_entropy = _entropy_from_counts(class_counts) if len(class_counts) else 0.0
    # Gini по классам: 1 - sum p^2
    p = class_counts / class_counts.sum() if class_counts.sum() > 0 else np.array([1.0])
    class_gini = float(1.0 - np.sum(p**2))

    # статистика по числовым
    num_cols = [c for c, flag in zip(X.columns, is_cat) if not flag]
    num_stats = {"mean": [], "std": [], "skew": [], "kurt": []}
    zero_var = 0

    for c in num_cols:
        s = _safe_numeric(X[c])
        if s is None:
            continue
        s = s.astype(float)
        m = float(s.mean(skipna=True))
        sd = float(s.std(skipna=True))
        sk = float(s.skew(skipna=True))
        ku = float(s.kurt(skipna=True))
        num_stats["mean"].append(m)
        num_stats["std"].append(sd)
        num_stats["skew"].append(sk)
        num_stats["kurt"].append(ku)
        if np.isfinite(sd) and abs(sd) < 1e-12:
            zero_var += 1

    mean_feature_mean = float(np.nanmean(num_stats["mean"])) if num_stats["mean"] else 0.0
    mean_feature_std  = float(np.nanmean(num_stats["std"]))  if num_stats["std"]  else 0.0
    mean_skewness     = float(np.nanmean(num_stats["skew"])) if num_stats["skew"] else 0.0
    mean_kurtosis     = float(np.nanmean(num_stats["kurt"])) if num_stats["kurt"] else 0.0
    share_zero_var    = float(zero_var / max(1, len(num_cols)))

    # средняя попарная |corr| по числовым (Spearman для робастности)
    avg_abs_corr_num = 0.0
    if len(num_cols) >= 2:
        corr = X[num_cols].corr(method="spearman")
        iu = np.triu_indices_from(corr, k=1)
        vals = np.abs(corr.values[iu])
        vals = vals[np.isfinite(vals)]
        avg_abs_corr_num = float(vals.mean()) if vals.size else 0.0

    # структурные
    share_categorical = float(n_cat_features / max(1, n_features))

    nunique_per_feature = []
    const_feats = 0
    cat_cards = []
    for c, flag in zip(X.columns, is_cat):
        nun = int(X[c].nunique(dropna=True))
        nunique_per_feature.append(nun)
        if nun <= 1:
            const_feats += 1
        if flag:
            cat_cards.append(nun)

    share_constant_features = float(const_feats / max(1, n_features))
    avg_unique_values_per_feature = float(np.mean(nunique_per_feature)) if nunique_per_feature else 0.0
    ratio_num_to_cat = float(n_num_features / max(1, n_cat_features)) if n_cat_features > 0 else float("inf")
    avg_cat_cardinality = float(np.mean(cat_cards)) if cat_cards else 0.0

    return MetaFeatures(
        # базовые
        n_rows=n_rows,
        n_features=n_features,
        n_classes=n_classes,
        n_cat_features=n_cat_features,
        missing_ratio=missing_ratio,
        class_entropy=class_entropy,
        # статистические
        mean_feature_mean=mean_feature_mean,
        mean_feature_std=mean_feature_std,
        mean_skewness=mean_skewness,
        mean_kurtosis=mean_kurtosis,
        avg_abs_corr_num=avg_abs_corr_num,
        share_zero_var=share_zero_var,
        # структурные
        share_categorical=share_categorical,
        share_constant_features=share_constant_features,
        avg_unique_values_per_feature=avg_unique_values_per_feature,
        ratio_num_to_cat=ratio_num_to_cat,
        avg_cat_cardinality=avg_cat_cardinality,
        class_gini=class_gini,
    )

# --- загрузка OpenML -------------------------------------------------------

def load_openml_dataset(did: int) -> Tuple[pd.DataFrame, pd.Series]:
    import openml
    ds = openml.datasets.get_dataset(did, download_data=True)
    X, y, _, _ = ds.get_data(target=ds.default_target_attribute)
    # попробуем привести object-колонки к category
    for c in X.columns:
        if pd.api.types.is_object_dtype(X[c]):
            X[c] = X[c].astype("category")
    return X, pd.Series(y, name=ds.default_target_attribute)

# --- эксперимент с перестановками -----------------------------------------

def permute_rows(X: pd.DataFrame, y: pd.Series, random_state: int = 42):
    rng = np.random.default_rng(random_state)
    idx = np.arange(len(X))
    rng.shuffle(idx)
    return X.iloc[idx].reset_index(drop=True), y.iloc[idx].reset_index(drop=True)

def permute_columns(X: pd.DataFrame, random_state: int = 43):
    rng = np.random.default_rng(random_state)
    cols = X.columns.to_list()
    rng.shuffle(cols)
    return X[cols].copy()

def relabel_categories(X: pd.DataFrame, y: pd.Series, random_state: int = 44):
    # переставим метки в категориальных фичах и в таргете (если он категориальный)
    rng = np.random.default_rng(random_state)

    def _permute_cats(s: pd.Series) -> pd.Series:
        if pd.api.types.is_categorical_dtype(s) or pd.api.types.is_object_dtype(s) or pd.api.types.is_integer_dtype(s):
            # берём уникальные значения и переотображаем
            vals = pd.Series(s.unique())
            mapping = vals.sample(frac=1.0, random_state=int(rng.integers(0, 1e9))).reset_index(drop=True)
            mp = dict(zip(vals, mapping))
            return s.map(mp)
        return s

    X_new = X.copy()
    for c in X_new.columns:
        X_new[c] = _permute_cats(X_new[c])

    y_new = _permute_cats(y.copy())

    # вернём категории как category, чтобы не прилипали типы object
    for c in X_new.columns:
        if pd.api.types.is_object_dtype(X_new[c]):
            X_new[c] = X_new[c].astype("category")
    if pd.api.types.is_object_dtype(y_new):
        y_new = y_new.astype("category")

    return X_new, y_new
