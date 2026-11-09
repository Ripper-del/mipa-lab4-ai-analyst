"""Пояснення зміни метрики: розклад по сегментах рахує код, гіпотезу формулює LLM."""
import json

import numpy as np
import pandas as pd

from . import prompts
from .verify import check_report

LOW_VOLUME = 30  # сегменти з меншою кількістю рядків у періоді вважаємо шумом


def _metric_value(frame, agg, column, date_col=None):
    if agg == 'count':
        return float(len(frame))
    if agg == 'nunique':  # середня кількість унікальних значень за день (на кшталт DAU)
        return float(frame.groupby(date_col)[column].nunique().mean()) if len(frame) else 0.0
    if agg == 'sum':
        return float(frame[column].sum())
    if agg == 'mean':
        return float(frame[column].mean()) if len(frame) else float('nan')
    raise ValueError(f'Невідома агрегація: {agg}')


def decompose(df, date_col, before, after, agg='count', column=None, dims=(), query=None, min_rows=20):
    """Розкладає зміну метрики (в розрахунку на день) на внески сегментів.

    before / after: пари дат (початок, кінець) включно.
    count/sum нормуються на кількість днів періоду; nunique рахується як середнє унікальних за день (DAU-подібне); mean береться як є.
    """
    d = df.copy()
    d[date_col] = pd.to_datetime(d[date_col], errors='coerce')
    if query:
        d = d.query(query)
    pb = d[(d[date_col] >= before[0]) & (d[date_col] <= before[1])]
    pa = d[(d[date_col] >= after[0]) & (d[date_col] <= after[1])]
    if len(pb) < min_rows or len(pa) < min_rows:
        raise ValueError(f'Замало даних у періодах (до: {len(pb)}, після: {len(pa)} рядків).')
    days_b = (pd.Timestamp(before[1]) - pd.Timestamp(before[0])).days + 1
    days_a = (pd.Timestamp(after[1]) - pd.Timestamp(after[0])).days + 1
    norm = (lambda v, days: v / days) if agg in ('count', 'sum') else (lambda v, days: v)
    total_b = norm(_metric_value(pb, agg, column, date_col), days_b)
    total_a = norm(_metric_value(pa, agg, column, date_col), days_a)
    facts = {
        'metric': f'{agg}({column or "*"})' + (f' where {query}' if query else ''),
        'period_before': [str(before[0]), str(before[1])], 'period_after': [str(after[0]), str(after[1])],
        'per_day_before': round(total_b, 3), 'per_day_after': round(total_a, 3),
        'relative_change_pct': round((total_a / total_b - 1) * 100, 1) if total_b else None,
        'segments': {},
    }
    total_delta = total_a - total_b
    for dim in dims:
        rows = []
        for val in sorted(set(pb[dim].dropna().unique()) | set(pa[dim].dropna().unique()), key=str):
            sb, sa = pb[pb[dim] == val], pa[pa[dim] == val]
            if len(sb) + len(sa) < min_rows:
                continue
            vb = norm(_metric_value(sb, agg, column, date_col), days_b) if len(sb) else 0.0
            va = norm(_metric_value(sa, agg, column, date_col), days_a) if len(sa) else 0.0
            rows.append({'segment': str(val), 'rows_before': int(len(sb)), 'rows_after': int(len(sa)),
                         'low_volume': bool(min(len(sb), len(sa)) < LOW_VOLUME), 'per_day_before': round(vb, 3), 'per_day_after': round(va, 3),
                         'change': round(va - vb, 3),
                         'relative_change_pct': round((va / vb - 1) * 100, 1) if vb else None,
                         'share_of_total_change_pct': round((va - vb) / total_delta * 100, 1) if total_delta and agg in ('count', 'sum', 'nunique') else None})
        rows.sort(key=lambda r: -abs(r['change']))
        facts['segments'][dim] = rows
    facts['concentration'] = _concentration(facts)
    return facts


def _concentration(facts):
    """Чи зміна зосереджена в одному сегменті, чи розмазана рівномірно."""
    out = {}
    for dim, rows in facts['segments'].items():
        rel = [r['relative_change_pct'] for r in rows if r['relative_change_pct'] is not None and not r['low_volume']]
        if len(rel) < 2:
            out[dim] = {'verdict': 'insufficient_data',
                        'note': 'надто мало сегментів із достатнім обсягом, висновку про рівномірність немає'}
            continue
        spread = float(np.std(rel))
        mean_rel = float(np.mean(rel))
        verdict = 'uniform' if spread < max(0.25 * abs(mean_rel), 5) else 'concentrated_or_mixed'
        out[dim] = {'verdict': verdict, 'mean_relative_change_pct': round(mean_rel, 1),
                    'std_relative_change_pct': round(spread, 1), 'largest_segment': rows[0]['segment']}
    return out


DROP_SCHEMA = {
    'type': 'object',
    'properties': {
        'summary': {'type': 'string'},
        'main_drivers': {'type': 'array', 'items': {'type': 'string'}},
        'hypothesis': {'type': 'string'},
        'checks': {'type': 'array', 'items': {'type': 'string'}},
    },
    'required': ['summary', 'main_drivers', 'hypothesis', 'checks'],
}


def explain_metric_drop(df, llm, date_col, before, after, dims, agg='count', column=None, query=None):
    facts = decompose(df, date_col, before, after, agg, column, dims, query)
    result = {'facts': facts, 'report': None}
    if llm is not None:
        user = 'ФАКТИ:\n' + json.dumps(facts, ensure_ascii=False, default=str)
        result['report'] = check_report(llm.structured(prompts.EXPLAIN_DROP_SYSTEM, user, DROP_SCHEMA, 'drop_report'), facts)
    return result
