"""Natural language -> Pandas. Код генерує LLM, перевіряє AST-фільтр, виконується в обмеженому оточенні."""
import ast

import numpy as np
import pandas as pd

from . import prompts

BANNED_NAMES = {'open', 'exec', 'eval', 'compile', '__import__', 'input', 'globals', 'locals',
                'getattr', 'setattr', 'delattr', 'vars', 'dir', 'breakpoint', 'exit', 'quit'}
BANNED_ATTRS = {'to_csv', 'to_excel', 'to_pickle', 'to_sql', 'to_parquet', 'to_json', 'to_clipboard',
                'read_csv', 'read_excel', 'read_pickle', 'read_sql', 'read_json', 'eval', 'system', 'popen'}
SAFE_BUILTINS = {n: __builtins__[n] if isinstance(__builtins__, dict) else getattr(__builtins__, n)
                 for n in ['len', 'sum', 'min', 'max', 'round', 'sorted', 'range', 'abs', 'list', 'dict',
                           'set', 'str', 'int', 'float', 'bool', 'tuple', 'enumerate', 'zip', 'any', 'all']}


class UnsafeCode(ValueError):
    pass


def check_code_safety(code):
    """Відхиляє import, файли, dunder-атрибути та небезпечні виклики."""
    tree = ast.parse(code)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            raise UnsafeCode('import заборонений')
        if isinstance(node, ast.Name) and node.id in BANNED_NAMES:
            raise UnsafeCode(f'виклик {node.id} заборонений')
        if isinstance(node, ast.Attribute) and (node.attr.startswith('_') or node.attr in BANNED_ATTRS):
            raise UnsafeCode(f'атрибут {node.attr} заборонений')
    return tree


def schema_text(df, max_values=6):
    """Схема для промпту: колонки, типи і приклади значень (без усього датасету)."""
    lines = []
    for c in df.columns:
        s = df[c]
        if pd.api.types.is_numeric_dtype(s):
            ex = f'числа від {s.min()} до {s.max()}'
        else:
            ex = 'значення: ' + ', '.join(map(repr, s.dropna().astype(str).unique()[:max_values]))
        lines.append(f'- {c} ({s.dtype}): {ex}')
    return '\n'.join(lines)


NL_SCHEMA = {
    'type': 'object',
    'properties': {'code': {'type': 'string', 'description': 'Python-код Pandas, результат у змінній result'},
                   'explanation': {'type': 'string', 'description': 'Що робить код, 1-2 речення'}},
    'required': ['code', 'explanation'],
}


def run_code(code, df):
    tree = check_code_safety(code)
    env = {'df': df.copy(), 'pd': pd, 'np': np}
    exec(compile(tree, '<llm>', 'exec'), {'__builtins__': SAFE_BUILTINS, 'pd': pd, 'np': np}, env)
    if 'result' not in env:
        raise ValueError('Код не записав результат у змінну result.')
    return env['result']


def ask(df, question, llm, retries=1):
    """Повертає {code, explanation, result, error}. Код завжди показується разом із результатом."""
    if df is None or len(df) == 0:
        raise ValueError('Порожня таблиця: немає до чого ставити питання.')
    if not question or not question.strip():
        raise ValueError('Порожнє питання.')
    user = f'СХЕМА df ({len(df)} рядків):\n{schema_text(df)}\n\nПИТАННЯ: {question.strip()}'
    error_note = ''
    for attempt in range(retries + 1):
        try:
            out = llm.structured(prompts.NL2PANDAS_SYSTEM, user + error_note, NL_SCHEMA, 'pandas_code')
        except RuntimeError as exc:  # модель відповіла не за схемою (типово для питань, на які немає відповіді в даних)
            return {'code': '', 'explanation': '', 'result': None,
                    'error': f'Модель не повернула код (можливо, питання не стосується даних): {exc}'}
        code = (out.get('code') or '').strip()
        if not code:
            return {'code': '', 'explanation': out.get('explanation', ''), 'result': None,
                    'error': 'Модель не змогла відповісти на це питання за наявними даними.'}
        try:
            result = run_code(code, df)
            return {'code': code, 'explanation': out.get('explanation', ''), 'result': result, 'error': None}
        except Exception as exc:  # самокорекція: віддаємо помилку моделі ще раз
            error_note = f'\n\nПОПЕРЕДНІЙ КОД:\n{code}\nПОМИЛКА: {type(exc).__name__}: {exc}\nВиправ код.'
            last = {'code': code, 'explanation': out.get('explanation', ''), 'result': None,
                    'error': f'{type(exc).__name__}: {exc}'}
    return last
