"""Готує плоску таблицю подій для демо з даних лабораторної №2 (events + users, очищені).

Запуск: python demo/prepare_demo_data.py <папка_з_CSV_лаби_2> demo/events_flat.csv
"""
import sys
from pathlib import Path

import pandas as pd

COUNTRY_MAP = {'Україна': 'UA', 'Ukraine': 'UA', 'ua': 'UA', 'Poland': 'PL', 'Germany': 'DE',
               'United States': 'US', 'USA': 'US', 'United Kingdom': 'GB', 'UK': 'GB'}
DATA_END = pd.Timestamp('2026-08-05')


def parse_dates(s):
    iso = pd.to_datetime(s.str.replace('/', '-'), format='ISO8601', errors='coerce')
    return iso.fillna(pd.to_datetime(s, format='%d.%m.%Y', errors='coerce'))


def main(src, dst):
    src = Path(src)
    users = pd.read_csv(src / 'users.csv', dtype=str).drop_duplicates()
    users['user_id'] = users['user_id'].astype(int)
    users['country'] = users['country'].str.strip().replace(COUNTRY_MAP).fillna('unknown')
    users['platform'] = users['platform'].str.strip().str.lower().map({'ios': 'iOS', 'android': 'Android'}).fillna('unknown')
    users['channel'] = users['marketing_channel'].str.strip().fillna('unknown')

    ev = pd.read_csv(src / 'events.csv').dropna(subset=['user_id']).drop_duplicates()
    ev['user_id'] = ev['user_id'].astype(int)
    ev['event_name'] = ev['event_name'].str.strip().str.lower().str.replace(' ', '_')
    ev['event_ts'] = parse_dates(ev['event_timestamp'])
    ev['event_date'] = ev['event_ts'].dt.normalize()
    ev = ev[ev['event_date'] <= DATA_END]
    flat = ev.merge(users[['user_id', 'country', 'platform', 'channel']], on='user_id', how='inner')
    flat['content_type'] = flat['event_params'].str.extract(r'content_type=(\w+)')
    flat = flat[['user_id', 'event_ts', 'event_date', 'event_name', 'content_type', 'country', 'platform', 'channel']]
    flat.to_csv(dst, index=False)
    print(f'{len(flat):,} рядків записано в {dst}')


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
