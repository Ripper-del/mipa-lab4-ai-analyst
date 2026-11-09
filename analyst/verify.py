"""Перевірка галюцинацій: чи є кожне число з відповіді моделі серед чисел, які порахував код."""
import json
import re

NUM_RE = re.compile(r'(?<![\w.])-?\d+(?:[  ]\d{3})*(?:[.,]\d+)?')


def _flatten_numbers(obj, out):
    if isinstance(obj, bool):
        return
    if isinstance(obj, (int, float)):
        out.add(float(obj))
    elif isinstance(obj, dict):
        for k, v in obj.items():
            _flatten_numbers(k, out)
            _flatten_numbers(v, out)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            _flatten_numbers(v, out)
    elif isinstance(obj, str):
        for m in NUM_RE.findall(obj):
            out.add(_to_float(m))


def _to_float(token):
    return float(token.replace(' ', '').replace(' ', '').replace(',', '.'))


def _close(x, known):
    x = abs(x)  # у тексті «впав на 3.3» і в фактах «-3.3» це те саме число
    for k in known:
        # дозволяємо округлення та перехід між частками і відсотками (0.19 <-> 19)
        for kk in (abs(k), abs(k) * 100, abs(k) / 100):
            if abs(x - kk) <= max(0.006 * kk, 0.051):
                return True
    return False


def unverified_numbers(text, facts):
    """Повертає числа з тексту, яких немає у фактах (з урахуванням округлення)."""
    known = set()
    _flatten_numbers(facts, known)
    bad = []
    for token in NUM_RE.findall(text):
        value = _to_float(token)
        if abs(value) < 10 and float(value).is_integer():
            continue  # дрібні лічильники («2 сегменти», «3 кроки») не перевіряємо
        if not _close(value, known):
            bad.append(token)
    return sorted(set(bad))


def check_report(report, facts):
    """Збирає всі рядки структури, перевіряє числа й додає поле `unverified_numbers`."""
    texts = []

    def walk(o):
        if isinstance(o, str):
            texts.append(o)
        elif isinstance(o, dict):
            for v in o.values():
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)
    walk(report)
    report = dict(report)
    report['unverified_numbers'] = unverified_numbers(' '.join(texts), facts)
    return report


def to_json(obj):
    return json.dumps(obj, ensure_ascii=False, default=str, indent=2)
