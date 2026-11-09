# AI-асистент аналітика

Плешу А. В., ІО-36. Лабораторна №4. Короткі технічні нотатки.

**Що це.** CLI на Python (`python -m analyst`). Команди: `eda`, `hypotheses`, `ask`, `drop`, `summary`.

**Принцип.** Числа рахує Pandas, модель лише інтерпретує вже порахований результат. Кожне число з відповіді моделі перевіряється за порахованими фактами (`analyst/verify.py`), непідтверджені позначаються `[УВАГА]`. Код від моделі для `ask` проходить AST-фільтр (без `import`, `open`, `eval`, `exec`) перед виконанням.

**Запуск.**
```bash
pip install -r requirements.txt
export OPENROUTER_API_KEY=...   # або GROQ_API_KEY / MISTRAL_API_KEY / GEMINI_API_KEY / ANTHROPIC_API_KEY
python -m analyst eda demo/events_flat.csv
python -m analyst --no-llm hypotheses demo/events_flat.csv --date-col event_date --user-col user_id
```
Прапорець `--no-llm` вимикає модель (працюють `eda`, `hypotheses`, `drop`). Провайдер обирається за наявним ключем або `--provider`.

**Структура.** `analyst/`: код (`llm.py` клієнти, `eda.py`, `hypotheses.py`, `nl2pandas.py`, `explain_drop.py`, `summary.py`, `verify.py`). `tests/`: тести (`python -m unittest discover tests`). `demo/`: дані, прогони й скриншоти.
