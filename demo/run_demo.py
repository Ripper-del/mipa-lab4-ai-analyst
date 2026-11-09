"""Демо на даних лабораторної №2 (події мобільного застосунку).

  python demo/run_demo.py            # з моделлю (потрібен GEMINI_API_KEY або ANTHROPIC_API_KEY), пише demo/demo_output.md
  python demo/run_demo.py --no-llm   # лише розрахункова частина, пише demo/demo_output_no_llm.md
"""
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from analyst import auto_eda, generate_hypotheses, ask, explain_metric_drop, executive_summary, collect_facts
from analyst.llm import LLMUnavailable, make_llm
from analyst.verify import to_json

NO_LLM = '--no-llm' in sys.argv
df = pd.read_csv(ROOT / 'demo' / 'events_flat.csv')
llm = None
if not NO_LLM:
    try:
        llm = make_llm()
    except LLMUnavailable as exc:
        sys.exit(f'{exc}\n(Для запуску без моделі: python demo/run_demo.py --no-llm)')

out = ['# Демо AI-асистента аналітика',
       f'\nДатасет: `demo/events_flat.csv`, {len(df):,} рядків (події застосунку з лаби №2). '
       f'Режим: **{"тільки розрахунки (без LLM)" if NO_LLM else "з LLM"}**.\n']


def step(title, fn, lang=''):
    """Крок із моделлю: збій одного кроку (ліміти безкоштовного тарифу, обрив мережі) не зупиняє демо."""
    try:
        block(title, fn(), lang)
    except Exception as exc:  # noqa: BLE001
        print(f'[{title}] помилка: {exc}')
        block(title, f'Крок не виконано: {type(exc).__name__}: {exc}')


def block(title, body, lang=''):
    out.append(f'\n## {title}\n')
    out.append(f'```{lang}\n{body}\n```' if lang else body)


# 1. EDA
profile = auto_eda(df)['profile']
block('1. Автоматичний EDA: профіль (порахував Pandas)', to_json(profile), 'json')
if llm:
    step('1. Автоматичний EDA: інтерпретація LLM', lambda: to_json(auto_eda(df, llm)['interpretation']), 'json')

# 2. Гіпотези
if NO_LLM:
    block('2. Факти для гіпотез (порахував код)', to_json(collect_facts(df, 'event_date', 'user_id')), 'json')
else:
    block('2. Факти, передані моделі', to_json(collect_facts(df, 'event_date', 'user_id')), 'json')
    step('2. Гіпотези', lambda: to_json(generate_hypotheses(df, llm, 'event_date', 'user_id')['report']), 'json')

# 3. NL -> Pandas
if not NO_LLM:
    def nl_step(q):
        res = ask(df, q, llm)
        body = f'Питання: {q}\n\nКод:\n```python\n{res["code"] or "(не згенеровано)"}\n```\n\nПояснення: {res["explanation"]}\n\n'
        body += f'Помилка: {res["error"]}' if res['error'] else f'Результат:\n```\n{res["result"].to_string() if hasattr(res["result"], "to_string") else res["result"]}\n```'
        return body

    for q in ['Покажи топ-5 країн за кількістю унікальних користувачів, які купили підписку',
              'Скільки в середньому подій view_content на користувача в кожному місяці?',
              'Яка погода завтра у Києві?']:
        step('3. Natural language -> Pandas', lambda q=q: nl_step(q))

# 4. Пояснення зміни метрики
DROP_ARGS = dict(date_col='event_date', before=('2026-03-01', '2026-05-31'), after=('2026-06-01', '2026-08-05'),
                 dims=['channel', 'platform', 'country'], agg='count', query="event_name == 'view_content'")
drop = explain_metric_drop(df, None, **DROP_ARGS)
block('4. Пояснення зміни метрики: розклад по сегментах (порахував код)', to_json(drop['facts']), 'json')
if llm:
    step('4. Пояснення зміни метрики: висновок LLM',
         lambda: to_json(explain_metric_drop(df, llm, **DROP_ARGS)['report']), 'json')

# 5. Executive summary
if llm:
    facts_all = collect_facts(df, 'event_date', 'user_id')
    metrics = {'drop': {k: drop['facts'][k] for k in ['metric', 'per_day_before', 'per_day_after', 'relative_change_pct']},
               'time': facts_all['time'], 'cohort_compare': facts_all['cohort_compare']}
    step('5. Executive summary', lambda: to_json(executive_summary(metrics, llm)), 'json')

name = 'demo_output_no_llm.md' if NO_LLM else 'demo_output.md'
(ROOT / 'demo' / name).write_text('\n'.join(out), encoding='utf-8')
print(f'Готово: demo/{name}')
