"""Автоматичний EDA. Усі числа рахує Pandas, LLM лише пише текстову інтерпретацію."""
import re

import numpy as np
import pandas as pd

from . import prompts
from .verify import check_report

DATE_PATTERN_LIMIT = 6


def _date_formats(series):
    """Шаблони значень рядкової колонки, що схожа на дату (цифри -> 9)."""
    pats = series.dropna().astype(str).str.replace(r'\d', '9', regex=True).value_counts()
    return pats


def _looks_like_date(series):
    sample = series.dropna().astype(str).head(200)
    if sample.empty:
        return False
    return sample.str.match(r'^\d{1,4}[-./]\d{1,2}[-./]\d{1,4}').mean() > 0.8


def _category_variants(series):
    """Значення, які відрізняються лише регістром/пробілами/розділювачами."""
    vals = series.dropna().astype(str).unique()
    groups = {}
    for v in vals:
        key = re.sub(r'[\s_\-]+', '_', v.strip().lower())
        groups.setdefault(key, []).append(v)
    return [sorted(g) for g in groups.values() if len(g) > 1]


def profile_dataframe(df):
    """Структурований профіль датасету: типи, пропуски, дублікати, розподіли, аномалії."""
    n = len(df)
    profile = {
        'rows': n,
        'columns': df.shape[1],
        'duplicate_rows': int(df.duplicated().sum()),
        'columns_info': {},
        'quality_flags': [],
    }
    for col in df.columns:
        s = df[col]
        info = {'dtype': str(s.dtype), 'missing': int(s.isna().sum()),
                'missing_pct': round(float(s.isna().mean() * 100), 2), 'unique': int(s.nunique())}
        if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
            d = s.dropna()
            if len(d):
                q1, q3 = d.quantile(0.25), d.quantile(0.75)
                iqr = q3 - q1
                outliers = int(((d < q1 - 3 * iqr) | (d > q3 + 3 * iqr)).sum()) if iqr > 0 else 0
                info.update(min=float(d.min()), max=float(d.max()), mean=round(float(d.mean()), 3),
                            median=float(d.median()), p99=float(d.quantile(0.99)), extreme_outliers=outliers)
                neg = int((d < 0).sum())
                if neg:
                    profile['quality_flags'].append(
                        {'column': col, 'issue': 'negative_values', 'count': neg})
        else:
            sv = s.dropna().astype(str)
            if _looks_like_date(s):
                pats = _date_formats(s)
                info['date_format_masks'] = {k: int(v) for k, v in pats.head(DATE_PATTERN_LIMIT).items()}
                info['date_format_masks_note'] = 'маска формату: кожна цифра замінена на 9, це НЕ значення дати і не проблема'
                if len(pats) > 1:
                    profile['quality_flags'].append(
                        {'column': col, 'issue': 'mixed_date_formats', 'count': int(pats.iloc[1:].sum())})
            elif s.nunique() <= 30:
                info['top_values'] = {str(k): int(v) for k, v in s.value_counts().head(8).items()}
            else:
                info['top_values'] = {str(k): int(v) for k, v in s.value_counts().head(3).items()}
            variants = _category_variants(s)
            if variants and s.nunique() <= 60:
                profile['quality_flags'].append(
                    {'column': col, 'issue': 'inconsistent_spelling', 'groups': variants[:6]})
            ws = int((sv != sv.str.strip()).sum())
            if ws:
                profile['quality_flags'].append({'column': col, 'issue': 'leading_trailing_spaces', 'count': ws})
        if info['missing_pct'] >= 5:
            profile['quality_flags'].append({'column': col, 'issue': 'many_missing', 'pct': info['missing_pct']})
        if info['unique'] <= 1 and n > 1:
            profile['quality_flags'].append({'column': col, 'issue': 'constant_column'})
        profile['columns_info'][col] = info
    if profile['duplicate_rows']:
        profile['quality_flags'].append({'column': '*', 'issue': 'duplicate_rows', 'count': profile['duplicate_rows']})
    return profile


EDA_SCHEMA = {
    'type': 'object',
    'properties': {
        'summary': {'type': 'string', 'description': 'Що це за датасет і загальна якість, 2-3 речення'},
        'quality_issues': {'type': 'array', 'items': {'type': 'object', 'properties': {
            'column': {'type': 'string'}, 'problem': {'type': 'string'}, 'suggested_fix': {'type': 'string'}},
            'required': ['column', 'problem', 'suggested_fix']}},
        'first_insights': {'type': 'array', 'items': {'type': 'string'}},
    },
    'required': ['summary', 'quality_issues', 'first_insights'],
}


def auto_eda(df, llm=None):
    """Профіль (завжди) + інтерпретація від LLM (якщо передано llm)."""
    if df is None or len(df) == 0:
        raise ValueError('Порожня таблиця: немає чого аналізувати.')
    profile = profile_dataframe(df)
    result = {'profile': profile, 'interpretation': None}
    if llm is not None:
        import json
        user = 'ФАКТИ (профіль датасету):\n' + json.dumps(profile, ensure_ascii=False, default=str)
        report = llm.structured(prompts.EDA_SYSTEM, user, EDA_SCHEMA, 'eda_report')
        result['interpretation'] = check_report(report, profile)
    return result
