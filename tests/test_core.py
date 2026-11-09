"""Тести без звернення до API: LLM замінено заглушкою. Запуск: python -m unittest discover tests"""
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from analyst.eda import auto_eda, profile_dataframe
from analyst.explain_drop import decompose, explain_metric_drop
from analyst.llm import OpenAICompatLLM, GeminiLLM, LLMUnavailable, make_llm, parse_json_object
from analyst.hypotheses import collect_facts, generate_hypotheses
from analyst.nl2pandas import UnsafeCode, ask, check_code_safety
from analyst.summary import executive_summary
from analyst.verify import unverified_numbers


class FakeLLM:
    """Повертає заздалегідь задані структури, ніби це відповідь моделі."""
    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = []

    def structured(self, system, user, schema, tool_name='x', max_tokens=0):
        self.calls.append((tool_name, user))
        return self.responses.pop(0)


def sample_df(n=400):
    rng = np.random.default_rng(1)
    dates = pd.date_range('2026-01-01', periods=120)
    d = pd.DataFrame({
        'user_id': rng.integers(1, 100, n),
        'day': rng.choice(dates.astype(str), n),
        'country': rng.choice(['UA', 'PL', 'DE', 'ua', ' PL'], n),
        'amount': rng.normal(50, 10, n).round(2),
    })
    d.loc[:4, 'amount'] = -5
    return pd.concat([d, d.iloc[:3]], ignore_index=True)


class TestEDA(unittest.TestCase):
    def test_profile_finds_issues(self):
        p = profile_dataframe(sample_df())
        issues = {(f['column'], f['issue']) for f in p['quality_flags']}
        self.assertIn(('amount', 'negative_values'), issues)
        self.assertIn(('country', 'inconsistent_spelling'), issues)
        self.assertIn(('country', 'leading_trailing_spaces'), issues)
        self.assertEqual(p['duplicate_rows'], 3)

    def test_empty_dataframe(self):
        with self.assertRaises(ValueError):
            auto_eda(pd.DataFrame())

    def test_auto_eda_flags_hallucinated_number(self):
        fake = FakeLLM({'summary': 'У таблиці 403 рядки і 987654 дублікатів.', 'quality_issues': [], 'first_insights': []})
        res = auto_eda(sample_df(), fake)
        self.assertIn('987654', res['interpretation']['unverified_numbers'])
        self.assertNotIn('403', res['interpretation']['unverified_numbers'])


class TestVerify(unittest.TestCase):
    def test_rounding_and_percent(self):
        facts = {'share': 0.1905, 'count': 1811, 'ratio': 2.8}
        self.assertEqual(unverified_numbers('Частка 19%, 1 811 користувачів, у 2,8 раза', facts), [])
        self.assertEqual(unverified_numbers('Впав на 3.3 п. п.', {'shift_pp': -3.3}), [])
        self.assertEqual(unverified_numbers('Частка 40%', facts), ['40'])


class TestNL2Pandas(unittest.TestCase):
    def test_runs_safe_code(self):
        df = sample_df()
        fake = FakeLLM({'code': "result = df.groupby('country')['amount'].mean()", 'explanation': 'середнє'})
        out = ask(df, 'середня сума по країнах', fake)
        self.assertIsNone(out['error'])
        self.assertIn('amount', out['code'])
        self.assertGreater(len(out['result']), 0)

    def test_blocks_unsafe_code(self):
        for bad in ["import os\nresult = 1", "result = open('/etc/passwd').read()",
                    "result = df.to_csv('x.csv')", "result = df.__class__"]:
            with self.assertRaises(UnsafeCode):
                check_code_safety(bad)

    def test_unsafe_code_from_model_is_reported_not_executed(self):
        fake = FakeLLM({'code': "import os\nresult = os.listdir('.')", 'explanation': ''},
                       {'code': "import os\nresult = 1", 'explanation': ''})
        out = ask(sample_df(), 'покажи файли', fake)
        self.assertIsNotNone(out['error'])
        self.assertIsNone(out['result'])

    def test_self_correction(self):
        fake = FakeLLM({'code': "result = df['нема_колонки'].sum()", 'explanation': ''},
                       {'code': "result = df['amount'].sum()", 'explanation': 'сума'})
        out = ask(sample_df(), 'сума', fake)
        self.assertIsNone(out['error'])
        self.assertIn('ПОМИЛКА', fake.calls[1][1])

    def test_illogical_question(self):
        fake = FakeLLM({'code': '', 'explanation': 'Немає такої колонки'})
        out = ask(sample_df(), 'яка погода завтра', fake)
        self.assertEqual(out['code'], '')
        self.assertIsNotNone(out['error'])

    def test_empty_inputs(self):
        with self.assertRaises(ValueError):
            ask(pd.DataFrame(), 'q', FakeLLM())
        with self.assertRaises(ValueError):
            ask(sample_df(), '  ', FakeLLM())


class TestDrop(unittest.TestCase):
    def make(self):
        rng = np.random.default_rng(0)
        rows = []
        for day in pd.date_range('2026-01-01', '2026-02-28'):
            rate = 100 if day < pd.Timestamp('2026-02-01') else 50
            for ch, share in [('a', 0.5), ('b', 0.5)]:
                rows += [{'day': day, 'channel': ch, 'user_id': i} for i in range(int(rate * share))]
        return pd.DataFrame(rows)

    def test_uniform_drop_detected(self):
        f = decompose(self.make(), 'day', ('2026-01-01', '2026-01-31'), ('2026-02-01', '2026-02-28'), 'count', dims=['channel'])
        self.assertEqual(f['relative_change_pct'], -50.0)
        self.assertEqual(f['concentration']['channel']['verdict'], 'uniform')

    def test_too_little_data(self):
        with self.assertRaises(ValueError):
            decompose(self.make(), 'day', ('2025-01-01', '2025-01-02'), ('2026-02-01', '2026-02-28'), dims=['channel'])

    def test_without_llm(self):
        r = explain_metric_drop(self.make(), None, 'day', ('2026-01-01', '2026-01-31'), ('2026-02-01', '2026-02-28'), ['channel'])
        self.assertIsNone(r['report'])


class TestHypothesesSummary(unittest.TestCase):
    def test_facts_and_generation(self):
        df = sample_df()
        facts = collect_facts(df, date_col='day', user_col='user_id')
        self.assertTrue(facts['rows'] > 0)
        fake = FakeLLM({'hypotheses': [{'hypothesis': 'h', 'evidence': f"рядків {facts['rows']}", 'how_to_check': 'x', 'confidence': 'low'}]})
        out = generate_hypotheses(df, fake, 'day', 'user_id')
        self.assertEqual(out['report']['unverified_numbers'], [])

    def test_summary_requires_metrics(self):
        with self.assertRaises(ValueError):
            executive_summary({}, FakeLLM())


class TestGemini(unittest.TestCase):
    def make(self, replies):
        llm = GeminiLLM(model='test-model', api_key='k', retries=1)
        llm.sent = []
        def fake_post(url, body):
            llm.sent.append((url, body))
            return {'candidates': [{'content': {'parts': [{'text': replies.pop(0)}]}}]}
        llm._post = fake_post
        return llm

    def test_parses_json_in_markdown_fence(self):
        llm = self.make(['```json\n{"a": 1, "b": "x"}\n```'])
        out = llm.structured('sys', 'user', {'type': 'object', 'required': ['a', 'b']})
        self.assertEqual(out, {'a': 1, 'b': 'x'})
        url, body = llm.sent[0]
        self.assertIn('test-model', url)
        self.assertEqual(body['generationConfig']['responseMimeType'], 'application/json')

    def test_retries_on_missing_fields(self):
        llm = self.make(['{"a": 1}', '{"a": 1, "b": 2}'])
        out = llm.structured('s', 'u', {'type': 'object', 'required': ['a', 'b']})
        self.assertEqual(out['b'], 2)
        self.assertIn('некоректною', llm.sent[1][1]['contents'][0]['parts'][0]['text'])

    def test_gives_up_on_garbage(self):
        llm = self.make(['не json', 'теж не json'])
        with self.assertRaises(RuntimeError):
            llm.structured('s', 'u', {'type': 'object', 'required': ['a']})

    def test_no_key_message(self):
        import os
        saved = {k: os.environ.pop(k, None) for k in ('GEMINI_API_KEY', 'GOOGLE_API_KEY', 'ANTHROPIC_API_KEY')}
        try:
            with self.assertRaises(LLMUnavailable):
                make_llm()
        finally:
            for k, v in saved.items():
                if v is not None:
                    os.environ[k] = v

    def test_parse_picks_last_json_after_reasoning(self):
        raw = 'Думаю: {"x": 1}. Ось відповідь: {"a": 1, "b": 2} кінець'
        self.assertEqual(parse_json_object(raw, ['a', 'b']), {'a': 1, 'b': 2})

    def test_parse_truncated_raises(self):
        with self.assertRaises(ValueError):
            parse_json_object('{"hypotheses": [{"hypothesis": "обірвано', ['hypotheses'])

    def test_parse_json_object_rejects_array(self):
        with self.assertRaises(ValueError):
            parse_json_object('[1, 2]')


class TestOpenAICompat(unittest.TestCase):
    def test_structured_json(self):
        llm = OpenAICompatLLM('groq', api_key='k', retries=1)
        sent = []
        def fake(path, body=None):
            sent.append((path, body))
            return {'choices': [{'message': {'content': '{"a": 5}'}}]}
        llm._request = fake
        self.assertEqual(llm.structured('s', 'u', {'type': 'object', 'required': ['a']}), {'a': 5})
        self.assertEqual(sent[0][0], '/chat/completions')
        self.assertEqual(sent[0][1]['response_format'], {'type': 'json_object'})

    def test_provider_choice_by_key(self):
        import os
        keys = ('GEMINI_API_KEY', 'GOOGLE_API_KEY', 'GROQ_API_KEY', 'OPENROUTER_API_KEY', 'MISTRAL_API_KEY', 'ANTHROPIC_API_KEY')
        saved = {k: os.environ.pop(k, None) for k in keys}
        try:
            os.environ['GROQ_API_KEY'] = 'x'
            self.assertEqual(make_llm().provider, 'groq')
        finally:
            os.environ.pop('GROQ_API_KEY', None)
            for k, v in saved.items():
                if v is not None:
                    os.environ[k] = v


if __name__ == '__main__':
    unittest.main()
