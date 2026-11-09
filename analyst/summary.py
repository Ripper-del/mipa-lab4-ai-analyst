"""Executive summary для стейкхолдера за набором уже розрахованих метрик."""
import json

from . import prompts
from .verify import check_report

SUMMARY_SCHEMA = {
    'type': 'object',
    'properties': {
        'what_happened': {'type': 'string'},
        'why_it_matters': {'type': 'string'},
        'next_steps': {'type': 'array', 'items': {'type': 'string'}, 'maxItems': 3},
    },
    'required': ['what_happened', 'why_it_matters', 'next_steps'],
}


def executive_summary(metrics, llm, audience='product manager'):
    if not metrics:
        raise ValueError('Немає метрик для summary.')
    user = f'АУДИТОРІЯ: {audience}\nФАКТИ:\n' + json.dumps(metrics, ensure_ascii=False, default=str)
    return check_report(llm.structured(prompts.SUMMARY_SYSTEM, user, SUMMARY_SCHEMA, 'summary'), metrics)
