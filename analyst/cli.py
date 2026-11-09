"""CLI: python -m analyst <команда> дані.csv [опції]"""
import argparse
import json
import sys

import pandas as pd

from .eda import auto_eda
from .explain_drop import explain_metric_drop
from .hypotheses import generate_hypotheses, collect_facts
from .llm import LLMUnavailable, make_llm
from .nl2pandas import ask
from .summary import executive_summary
from .verify import to_json


def load(path):
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        sys.exit(f'Файл не знайдено: {path}')
    except pd.errors.EmptyDataError:
        sys.exit('Файл порожній.')
    return df


def get_llm(args):
    if getattr(args, 'no_llm', False):
        return None
    try:
        return make_llm(args.provider, args.model)
    except LLMUnavailable as exc:
        sys.exit(str(exc))


def require_llm(args):
    if args.no_llm:
        sys.exit('Ця команда потребує моделі: приберіть --no-llm і задайте GEMINI_API_KEY або ANTHROPIC_API_KEY.')
    return get_llm(args)


def print_flags(report):
    bad = report.get('unverified_numbers')
    if bad:
        print(f'\n[УВАГА] Ці числа з відповіді моделі не знайдені серед порахованих фактів: {bad}. Перевірте вручну.')
    elif bad is not None:
        print('\n[OK] Усі числа у відповіді моделі знайдені серед порахованих фактів.')


def cmd_eda(args):
    df = load(args.file)
    res = auto_eda(df, get_llm(args))
    print(to_json(res['profile']))
    if res['interpretation']:
        print('\n=== Інтерпретація LLM ===')
        print(to_json(res['interpretation']))
        print_flags(res['interpretation'])


def cmd_hypotheses(args):
    df = load(args.file)
    if args.no_llm:
        print(to_json(collect_facts(df, args.date_col, args.user_col)))
        return
    res = generate_hypotheses(df, get_llm(args), args.date_col, args.user_col)
    print('=== Факти, передані моделі ===')
    print(to_json(res['facts']))
    print('\n=== Гіпотези ===')
    for i, h in enumerate(res['report'].get('hypotheses', []), 1):
        print(f"{i}. {h['hypothesis']}\n   Докази: {h['evidence']}\n   Перевірка: {h['how_to_check']}\n   Впевненість: {h['confidence']}")
    if res['report'].get('note'):
        print(res['report']['note'])
    print_flags(res['report'])


def cmd_ask(args):
    df = load(args.file)
    out = ask(df, args.question, require_llm(args))
    print('=== Код (перевірте логіку) ===')
    print(out['code'] or '(код не згенеровано)')
    print('\n=== Пояснення ===')
    print(out['explanation'])
    print('\n=== Результат ===')
    if out['error']:
        print('Помилка:', out['error'])
    else:
        print(out['result'].to_string() if hasattr(out['result'], 'to_string') else out['result'])


def cmd_drop(args):
    df = load(args.file)
    res = explain_metric_drop(
        df, None if args.no_llm else get_llm(args), args.date_col, tuple(args.before), tuple(args.after),
        args.dims, args.agg, args.column, args.query)
    print('=== Розклад по сегментах (порахував код) ===')
    print(to_json(res['facts']))
    if res['report']:
        print('\n=== Пояснення LLM ===')
        print(to_json(res['report']))
        print_flags(res['report'])


def cmd_summary(args):
    with open(args.metrics, encoding='utf-8') as f:
        metrics = json.load(f)
    rep = executive_summary(metrics, require_llm(args), args.audience)
    print(to_json(rep))
    print_flags(rep)


def build_parser():
    p = argparse.ArgumentParser(prog='analyst', description='AI-асистент аналітика')
    p.add_argument('--provider', default='auto', choices=['auto', 'gemini', 'groq', 'openrouter', 'mistral', 'claude'], help='LLM-провайдер (за замовчуванням за наявним ключем)')
    p.add_argument('--model', default=None, help='ID моделі (інакше за замовчуванням провайдера або ANALYST_MODEL)')
    p.add_argument('--no-llm', action='store_true', help='лише розрахунки, без звернення до моделі')
    sub = p.add_subparsers(dest='cmd', required=True)

    s = sub.add_parser('eda'); s.add_argument('file'); s.set_defaults(fn=cmd_eda)
    s = sub.add_parser('hypotheses'); s.add_argument('file'); s.add_argument('--date-col'); s.add_argument('--user-col'); s.set_defaults(fn=cmd_hypotheses)
    s = sub.add_parser('ask'); s.add_argument('file'); s.add_argument('question'); s.set_defaults(fn=cmd_ask)
    s = sub.add_parser('drop'); s.add_argument('file'); s.add_argument('--date-col', required=True)
    s.add_argument('--before', nargs=2, required=True, metavar=('FROM', 'TO'))
    s.add_argument('--after', nargs=2, required=True, metavar=('FROM', 'TO'))
    s.add_argument('--dims', nargs='+', required=True)
    s.add_argument('--agg', default='count', choices=['count', 'nunique', 'sum', 'mean'])
    s.add_argument('--column'); s.add_argument('--query'); s.set_defaults(fn=cmd_drop)
    s = sub.add_parser('summary'); s.add_argument('metrics', help='JSON з розрахованими метриками'); s.add_argument('--audience', default='product manager'); s.set_defaults(fn=cmd_summary)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        args.fn(args)
    except ValueError as exc:
        sys.exit(f'Помилка: {exc}')


if __name__ == '__main__':
    main()
