"""Генерація гіпотез: код збирає перевіряємі факти, LLM формулює з них гіпотези."""
import itertools
import json

import numpy as np
import pandas as pd

from . import prompts
from .verify import check_report

MIN_GROUP = 30


def _weekly_series(df, date_col):
    d = pd.to_datetime(df[date_col], errors='coerce').dropna()
    if d.empty:
        return None
    return d.dt.to_period('W').dt.start_time.value_counts().sort_index()


def _level_shift(series, window=4):
    """Тиждень, після якого середній рівень ряду найбільше відрізняється від попередніх `window` тижнів."""
    best = None
    for i in range(window, len(series) - window + 1):
        before, after = series.iloc[i - window:i].mean(), series.iloc[i:i + window].mean()
        if before > 0 and after > 0:
            score = abs(np.log(after / before))
            if best is None or score > best[0]:
                best = (score, series.index[i], before, after)
    return best


def _cohort_compare(df, date_col, user_col, cat_cols, break_week):
    """Порівнює користувачів, які вперше з'явились до і після break_week."""
    d = pd.to_datetime(df[date_col], errors='coerce')
    work = df.assign(_d=d).dropna(subset=['_d'])
    first = work.groupby(user_col)['_d'].min().rename('_first')
    work = work.join(first, on=user_col)
    work['_cohort'] = np.where(work['_first'] >= break_week, 'after', 'before')
    data_end = work['_d'].max()
    out = {'break_week': str(break_week.date()), 'note': 'rows_first_14d_per_user рахується лише для користувачів, у яких є повні 14 днів спостереження'}
    for label in ['before', 'after']:
        part = work[work['_cohort'] == label]
        users = part[user_col].nunique()
        out[f'{label}_new_users'] = int(users)
        mature = part[part['_first'] + pd.Timedelta(days=14) <= data_end]
        early = mature[mature['_d'] < mature['_first'] + pd.Timedelta(days=14)]
        out[f'{label}_rows_first_14d_per_user'] = round(len(early) / max(mature[user_col].nunique(), 1), 2)
    shifts = []
    firsts = work.sort_values('_d').drop_duplicates(user_col)
    # порівнюємо структуру лише за атрибутами користувача (константні в межах користувача), а не подій
    user_dims = [c for c in cat_cols if (work.groupby(user_col)[c].nunique(dropna=True) <= 1).mean() >= 0.95]
    for dim in user_dims:
        sb = firsts.loc[firsts['_cohort'] == 'before', dim].value_counts(normalize=True)
        sa = firsts.loc[firsts['_cohort'] == 'after', dim].value_counts(normalize=True)
        for val in set(sb.index) | set(sa.index):
            b, a = float(sb.get(val, 0)), float(sa.get(val, 0))
            shifts.append({'dimension': dim, 'segment': str(val), 'share_before_pct': round(b * 100, 1),
                           'share_after_pct': round(a * 100, 1), 'shift_pp': round((a - b) * 100, 1)})
    out['largest_structure_shifts'] = sorted(shifts, key=lambda x: -abs(x['shift_pp']))[:5]
    return out


def collect_facts(df, date_col=None, user_col=None, max_dims=6, max_facts=25):
    """Факти для гіпотез: розриви в часі, зсуви середніх по сегментах, сильні кореляції."""
    facts = {'rows': len(df), 'time': None, 'segment_gaps': [], 'correlations': []}
    cat_cols = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c]) and 2 <= df[c].nunique() <= 20
                and c not in (date_col, user_col)][:max_dims]
    num_cols = [c for c in df.select_dtypes('number').columns if c != user_col and df[c].nunique() > 5]

    if date_col:
        dates = pd.to_datetime(df[date_col], errors='coerce')
        valid = df.assign(_d=dates).dropna(subset=['_d'])
        if user_col:
            first_seen = valid.groupby(user_col)['_d'].min()
            weekly, series_name = first_seen.dt.to_period('W').dt.start_time.value_counts().sort_index(), 'new_users_per_week'
        else:
            weekly, series_name = _weekly_series(df, date_col), 'rows_per_week'
        # перший і останній тиждень неповні, тому прибираємо їх
        core = weekly.iloc[1:-1] if weekly is not None and len(weekly) > 9 else None
        best = _level_shift(core) if core is not None else None
        if best:
            _, break_week, avg_b, avg_a = best
            facts['time'] = {'series': series_name, 'break_week': str(break_week.date()),
                             'avg_per_week_4w_before': round(float(avg_b), 1), 'avg_per_week_4w_after': round(float(avg_a), 1),
                             'change_ratio': round(float(avg_a / avg_b), 3)}
            if user_col:
                facts['cohort_compare'] = _cohort_compare(df, date_col, user_col, cat_cols, break_week)

    # зсуви середнього числової метрики між сегментами
    for dim in cat_cols:
        for num in num_cols[:4]:
            g = df.groupby(dim)[num].agg(['mean', 'size'])
            g = g[g['size'] >= MIN_GROUP]
            if len(g) < 2 or g['mean'].min() <= 0:
                continue
            hi, lo = g['mean'].idxmax(), g['mean'].idxmin()
            ratio = float(g.loc[hi, 'mean'] / g.loc[lo, 'mean'])
            facts['segment_gaps'].append({
                'dimension': dim, 'metric': num, 'highest_segment': str(hi), 'highest_mean': round(float(g.loc[hi, 'mean']), 3),
                'lowest_segment': str(lo), 'lowest_mean': round(float(g.loc[lo, 'mean']), 3), 'ratio': round(ratio, 3),
                'rows_highest': int(g.loc[hi, 'size']), 'rows_lowest': int(g.loc[lo, 'size'])})
    if user_col and user_col in df.columns:
        for dim in cat_cols:
            g = df.groupby(dim)[user_col].agg(rows='size', users='nunique')
            g = g[g['users'] >= MIN_GROUP]
            if len(g) >= 2:
                g['rows_per_user'] = g['rows'] / g['users']
                hi, lo = g['rows_per_user'].idxmax(), g['rows_per_user'].idxmin()
                facts['segment_gaps'].append({
                    'dimension': dim, 'metric': 'rows_per_user', 'highest_segment': str(hi),
                    'highest_mean': round(float(g.loc[hi, 'rows_per_user']), 3), 'lowest_segment': str(lo),
                    'lowest_mean': round(float(g.loc[lo, 'rows_per_user']), 3),
                    'ratio': round(float(g.loc[hi, 'rows_per_user'] / g.loc[lo, 'rows_per_user']), 3),
                    'rows_highest': int(g.loc[hi, 'users']), 'rows_lowest': int(g.loc[lo, 'users'])})
    facts['segment_gaps'] = sorted(facts['segment_gaps'], key=lambda f: -abs(f['ratio'] - 1))[:max_facts]

    if len(num_cols) >= 2:
        corr = df[num_cols].corr(numeric_only=True)
        for a, b in itertools.combinations(num_cols, 2):
            v = corr.loc[a, b]
            if pd.notna(v) and abs(v) >= 0.4:
                facts['correlations'].append({'pair': [a, b], 'pearson': round(float(v), 3)})
        facts['correlations'] = sorted(facts['correlations'], key=lambda f: -abs(f['pearson']))[:5]
    return facts


HYP_SCHEMA = {
    'type': 'object',
    'properties': {'hypotheses': {'type': 'array', 'minItems': 5, 'maxItems': 10, 'items': {
        'type': 'object',
        'properties': {
            'hypothesis': {'type': 'string'},
            'evidence': {'type': 'string', 'description': 'Числа з фактів, на які спирається гіпотеза'},
            'how_to_check': {'type': 'string'},
            'confidence': {'type': 'string', 'enum': ['low', 'medium', 'high']}},
        'required': ['hypothesis', 'evidence', 'how_to_check', 'confidence']}}},
    'required': ['hypotheses'],
}


def generate_hypotheses(df, llm, date_col=None, user_col=None, extra_facts=None):
    """5-10 гіпотез, кожна з числами з фактів. Числа, яких немає у фактах, позначаються."""
    if df is None or len(df) == 0:
        raise ValueError('Порожня таблиця: немає чого аналізувати.')
    facts = collect_facts(df, date_col, user_col)
    if extra_facts:
        facts['extra'] = extra_facts
    if not (facts['time'] or facts['segment_gaps'] or facts['correlations']):
        return {'facts': facts, 'report': {'hypotheses': [], 'unverified_numbers': [],
                                           'note': 'У даних не знайдено що порівнювати (немає дат, категорій чи числових колонок).'}}
    user = 'ФАКТИ:\n' + json.dumps(facts, ensure_ascii=False, default=str)
    report = llm.structured(prompts.HYPOTHESES_SYSTEM, user, HYP_SCHEMA, 'hypotheses')
    return {'facts': facts, 'report': check_report(report, facts)}
