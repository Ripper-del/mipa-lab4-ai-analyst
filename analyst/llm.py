"""Обгортки над LLM API. Усе, що звертається до моделі, йде тільки через цей файл.

Три клієнти з однаковим інтерфейсом `text(...)` і `structured(...)`:
  ClaudeLLM        Anthropic API (tool use для структурованого виходу)
  GeminiLLM        Google Gemini (REST)
  OpenAICompatLLM  Groq, OpenRouter, Mistral (OpenAI-сумісний /chat/completions)
"""
import json
import os
import time
import urllib.error
import urllib.request


class LLMUnavailable(RuntimeError):
    """Немає API-ключа або не встановлено бібліотеку."""


def http_json(url, headers, body=None, timeout=300, retries=4, retry_wait=30, name='API'):
    """JSON-запит із повтором при 429/5xx і мережевих таймаутах (безкоштовні тарифи мають ліміти)."""
    data = None if body is None else json.dumps(body).encode('utf-8')
    for attempt in range(retries):
        req = urllib.request.Request(url, data=data, method='GET' if body is None else 'POST', headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode('utf-8'))
        except urllib.error.HTTPError as exc:
            if exc.code in (429, 500, 502, 503) and attempt < retries - 1:
                time.sleep(retry_wait)
                continue
            detail = exc.read().decode('utf-8', 'ignore')[:300]
            raise RuntimeError(f'{name} {exc.code}: {detail}') from exc
        except OSError as exc:  # таймаут, обрив з'єднання, DNS (HTTPError оброблено вище)
            if attempt < retries - 1:
                continue
            raise RuntimeError(f'{name}: мережа або таймаут: {exc}') from exc


def parse_json_object(raw, required=()):
    """Дістає JSON-об'єкт з відповіді: з ```json ... ``` або з тексту, де модель «міркує вголос».

    Серед усіх JSON-об'єктів у тексті бере останній, який містить усі обов'язкові поля.
    """
    dec = json.JSONDecoder()
    found, i = [], 0
    while True:
        i = raw.find('{', i)
        if i == -1:
            break
        try:
            obj, end = dec.raw_decode(raw, i)
        except json.JSONDecodeError:
            i += 1
            continue
        if isinstance(obj, dict):
            found.append(obj)
        i = end
    if not found:
        raise ValueError("коректний JSON-об'єкт не знайдено (відповідь могла бути обірвана)")
    good = [o for o in found if all(k in o for k in required)]
    if not good:
        raise ValueError(f'немає полів: {[k for k in required if k not in found[-1]]}')
    return good[-1]


def json_prompt(user, schema):
    return (user + '\n\nПоверни ЛИШЕ валідний JSON-об\'єкт (без пояснень і без markdown), що відповідає цій '
            'JSON-схемі. Усі текстові значення пиши українською мовою (назви ключів не змінюй). '
            'Не пиши міркувань, тільки JSON:\n' + json.dumps(schema, ensure_ascii=False))


class _JsonModeMixin:
    """structured() для клієнтів без tool use: просимо JSON, парсимо, при невдачі один раз перепитуємо."""

    def structured(self, system, user, schema, tool_name='report', max_tokens=6000):
        prompt = json_prompt(user, schema)
        last_err = None
        for _ in range(2):
            raw = self._generate(system, prompt, max_tokens, json_mode=True)
            try:
                return parse_json_object(raw, schema.get('required', []))
            except ValueError as exc:
                last_err = exc
                prompt += f'\n\nПопередня відповідь була некоректною ({exc}). Поверни виправлений JSON.'
        raise RuntimeError(f'Модель не повернула коректний JSON: {last_err}')

    def text(self, system, user, max_tokens=1500):
        return self._generate(system, user, max_tokens, json_mode=False)


class ClaudeLLM:
    def __init__(self, model=None, api_key=None):
        key = api_key or os.environ.get('ANTHROPIC_API_KEY')
        if not key:
            raise LLMUnavailable('Не знайдено ANTHROPIC_API_KEY. Додайте ключ у змінну середовища '
                                 'або запустіть з прапорцем --no-llm.')
        try:
            import anthropic
        except ImportError as exc:
            raise LLMUnavailable('Встановіть залежності: pip install -r requirements.txt') from exc
        self.client = anthropic.Anthropic(api_key=key)
        self.model = model or os.environ.get('ANALYST_MODEL', 'claude-sonnet-5-5')
        self.provider = 'claude'

    def text(self, system, user, max_tokens=1500):
        resp = self.client.messages.create(
            model=self.model, max_tokens=max_tokens, system=system,
            messages=[{'role': 'user', 'content': user}])
        return ''.join(b.text for b in resp.content if b.type == 'text')

    def structured(self, system, user, schema, tool_name='report', max_tokens=2500):
        """Структурована відповідь через примусовий виклик tool: модель повертає JSON за схемою."""
        resp = self.client.messages.create(
            model=self.model, max_tokens=max_tokens, system=system,
            tools=[{'name': tool_name, 'description': 'Поверни результат у вказаній структурі.',
                    'input_schema': schema}],
            tool_choice={'type': 'tool', 'name': tool_name},
            messages=[{'role': 'user', 'content': user}])
        for block in resp.content:
            if block.type == 'tool_use':
                return block.input
        raise RuntimeError('Модель не повернула структурований результат.')


class GeminiLLM(_JsonModeMixin):
    """Google Gemini через REST, без SDK (ключ з Google AI Studio)."""
    ENDPOINT = 'https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent'

    def __init__(self, model=None, api_key=None, retries=4, retry_wait=30):
        key = api_key or os.environ.get('GEMINI_API_KEY') or os.environ.get('GOOGLE_API_KEY')
        if not key:
            raise LLMUnavailable('Не знайдено GEMINI_API_KEY. Ключ: aistudio.google.com -> Get API key.')
        self.provider, self.key = 'gemini', key
        self.model = model or os.environ.get('ANALYST_MODEL', 'gemini-3.8-flash')
        self.retries, self.retry_wait = retries, retry_wait

    def _post(self, url, body):
        return http_json(url, {'Content-Type': 'application/json', 'x-goog-api-key': self.key}, body,
                         retries=self.retries, retry_wait=self.retry_wait, name='Gemini API')

    def _generate(self, system, user, max_tokens, json_mode):
        body = {'systemInstruction': {'parts': [{'text': system}]},
                'contents': [{'role': 'user', 'parts': [{'text': user}]}],
                'generationConfig': {'maxOutputTokens': max_tokens, 'temperature': 0.2}}
        if json_mode:
            body['generationConfig']['responseMimeType'] = 'application/json'
        data = self._post(self.ENDPOINT.format(model=self.model), body)
        try:
            parts = data['candidates'][0]['content']['parts']
        except (KeyError, IndexError):
            raise RuntimeError(f'Gemini не повернув відповідь: {str(data)[:300]}')
        return ''.join(p.get('text', '') for p in parts)


class OpenAICompatLLM(_JsonModeMixin):
    """Будь-який сервіс із OpenAI-сумісним API: Groq, OpenRouter, Mistral (є безкоштовні ключі)."""
    PROVIDERS = {
        'groq': ('https://api.groq.com/openai/v1', 'GROQ_API_KEY', 'llama-3.3-70b-versatile'),
        'openrouter': ('https://openrouter.ai/api/v1', 'OPENROUTER_API_KEY', 'nvidia/nemotron-3-ultra-550b-a55b:free'),
        'mistral': ('https://api.mistral.ai/v1', 'MISTRAL_API_KEY', 'mistral-small-latest'),
    }

    def __init__(self, provider, model=None, api_key=None, retries=4, retry_wait=30):
        base, env, default_model = self.PROVIDERS[provider]
        key = api_key or os.environ.get(env)
        if not key:
            raise LLMUnavailable(f'Не знайдено {env}.')
        self.provider, self.base, self.key = provider, base, key
        self.model = model or os.environ.get('ANALYST_MODEL', default_model)
        self.retries, self.retry_wait = retries, retry_wait

    def _request(self, path, body=None):
        headers = {'Content-Type': 'application/json', 'Authorization': f'Bearer {self.key}', 'User-Agent': 'ai-analyst/1.0'}
        return http_json(self.base + path, headers, body, retries=self.retries, retry_wait=self.retry_wait,
                         name=f'{self.provider} API')

    def list_models(self):
        return sorted(m['id'] for m in self._request('/models')['data'])

    def _generate(self, system, user, max_tokens, json_mode):
        body = {'model': self.model, 'max_tokens': max_tokens, 'temperature': 0.2,
                'messages': [{'role': 'system', 'content': system}, {'role': 'user', 'content': user}]}
        if json_mode:
            body['response_format'] = {'type': 'json_object'}
        data = self._request('/chat/completions', body)
        try:
            return data['choices'][0]['message']['content'] or ''
        except (KeyError, IndexError):
            raise RuntimeError(f'{self.provider} не повернув відповідь: {str(data)[:300]}')


def make_llm(provider='auto', model=None):
    """Обирає провайдера: явно або за наявним ключем у змінних середовища."""
    if provider == 'auto':
        found = [name for name, env in [('groq', 'GROQ_API_KEY'), ('openrouter', 'OPENROUTER_API_KEY'),
                                        ('mistral', 'MISTRAL_API_KEY'), ('gemini', 'GEMINI_API_KEY'),
                                        ('gemini', 'GOOGLE_API_KEY'), ('claude', 'ANTHROPIC_API_KEY')]
                 if os.environ.get(env)]
        if not found:
            raise LLMUnavailable('Не знайдено ключа API. Задайте GROQ_API_KEY / OPENROUTER_API_KEY / MISTRAL_API_KEY / '
                                 'GEMINI_API_KEY (є безкоштовні варіанти) або ANTHROPIC_API_KEY, або запустіть з --no-llm.')
        provider = found[0]
    if provider == 'gemini':
        return GeminiLLM(model=model)
    if provider == 'claude':
        return ClaudeLLM(model=model)
    if provider in OpenAICompatLLM.PROVIDERS:
        return OpenAICompatLLM(provider, model=model)
    raise LLMUnavailable(f'Невідомий провайдер: {provider}')
