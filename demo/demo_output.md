# Демо AI-асистента аналітика

Датасет: `demo/events_flat.csv`, 41,997 рядків (події застосунку з лаби №2). Режим: **з LLM**.


## 1. Автоматичний EDA: профіль (порахував Pandas)

```json
{
  "rows": 41997,
  "columns": 8,
  "duplicate_rows": 3,
  "columns_info": {
    "user_id": {
      "dtype": "int64",
      "missing": 0,
      "missing_pct": 0.0,
      "unique": 4000,
      "min": 1000.0,
      "max": 4999.0,
      "mean": 2999.311,
      "median": 3005.0,
      "p99": 4958.0,
      "extreme_outliers": 0
    },
    "event_ts": {
      "dtype": "str",
      "missing": 0,
      "missing_pct": 0.0,
      "unique": 41503,
      "date_format_masks": {
        "9999-99-99 99:99:99": 41997
      },
      "date_format_masks_note": "маска формату: кожна цифра замінена на 9, це НЕ значення дати і не проблема"
    },
    "event_date": {
      "dtype": "str",
      "missing": 0,
      "missing_pct": 0.0,
      "unique": 217,
      "date_format_masks": {
        "9999-99-99": 41997
      },
      "date_format_masks_note": "маска формату: кожна цифра замінена на 9, це НЕ значення дати і не проблема"
    },
    "event_name": {
      "dtype": "str",
      "missing": 0,
      "missing_pct": 0.0,
      "unique": 6,
      "top_values": {
        "view_content": 24708,
        "open_app": 12267,
        "registration": 4001,
        "start_trial": 642,
        "purchase": 315,
        "cancel_subscription": 64
      }
    },
    "content_type": {
      "dtype": "str",
      "missing": 17289,
      "missing_pct": 41.17,
      "unique": 5,
      "top_values": {
        "course": 4994,
        "article": 4980,
        "video": 4973,
        "quiz": 4921,
        "podcast": 4840
      }
    },
    "country": {
      "dtype": "str",
      "missing": 0,
      "missing_pct": 0.0,
      "unique": 13,
      "top_values": {
        "UA": 13600,
        "PL": 5361,
        "DE": 4397,
        "US": 4154,
        "FR": 2762,
        "GB": 2579,
        "ES": 2182,
        "IT": 2131
      }
    },
    "platform": {
      "dtype": "str",
      "missing": 0,
      "missing_pct": 0.0,
      "unique": 3,
      "top_values": {
        "iOS": 21137,
        "Android": 20261,
        "unknown": 599
      }
    },
    "channel": {
      "dtype": "str",
      "missing": 0,
      "missing_pct": 0.0,
      "unique": 7,
      "top_values": {
        "organic": 12277,
        "google_ads": 9025,
        "facebook_ads": 7722,
        "referral": 5197,
        "email": 3540,
        "apple_search": 3032,
        "unknown": 1204
      }
    }
  },
  "quality_flags": [
    {
      "column": "content_type",
      "issue": "many_missing",
      "pct": 41.17
    },
    {
      "column": "*",
      "issue": "duplicate_rows",
      "count": 3
    }
  ]
}
```

## 1. Автоматичний EDA: інтерпретація LLM

```json
{
  "summary": "Датасет містить 41 997 подій від 4 000 унікальних користувачів за 217 днів. Якість даних задовільна: є 3 повні дублікати рядків і 41,17% пропусків у колонці content_type, інші колонки мають узгоджені формати дат і відсутні пропуски.",
  "quality_issues": [
    {
      "column": "content_type",
      "problem": "Високий відсоток пропущених значень (41,17%)",
      "suggested_fix": "Заповнити пропуски за допомогою моделювання або видалити рядки, якщо це не критично для аналізу; перевірити логування подій, де content_type не фіксується."
    },
    {
      "column": "*",
      "problem": "Наявність 3 повністю дубльованих рядків",
      "suggested_fix": "Видалити дублікати, залишивши унікальні рядки."
    }
  ],
  "first_insights": [
    "Найчастіші події: view_content (24 708) та open_app (12 267), покупки (purchase) лише 315.",
    "Контент типів course, article, video, quiz, podcast мають схожу частоту (~4 900 кожен), але 41% подій не мають content_type.",
    "Основні країни: UA (13 600), PL (5 361), DE (4 397).",
    "Платформи: iOS (21 137), Android (20 261), unknown (599).",
    "Канали: organic (12 277), google_ads (9 025), facebook_ads (7 722), unknown (1 204)."
  ],
  "unverified_numbers": []
}
```

## 2. Факти, передані моделі

```json
{
  "rows": 41997,
  "time": {
    "series": "new_users_per_week",
    "break_week": "2026-06-01",
    "avg_per_week_4w_before": 100.5,
    "avg_per_week_4w_after": 193.2,
    "change_ratio": 1.923
  },
  "segment_gaps": [
    {
      "dimension": "event_name",
      "metric": "rows_per_user",
      "highest_segment": "view_content",
      "highest_mean": 6.451,
      "lowest_segment": "cancel_subscription",
      "lowest_mean": 1.0,
      "ratio": 6.451,
      "rows_highest": 3830,
      "rows_lowest": 64
    },
    {
      "dimension": "country",
      "metric": "rows_per_user",
      "highest_segment": "unknown",
      "highest_mean": 12.237,
      "lowest_segment": "IN",
      "lowest_mean": 5.719,
      "ratio": 2.14,
      "rows_highest": 97,
      "rows_lowest": 57
    },
    {
      "dimension": "channel",
      "metric": "rows_per_user",
      "highest_segment": "unknown",
      "highest_mean": 10.945,
      "lowest_segment": "apple_search",
      "lowest_mean": 10.073,
      "ratio": 1.087,
      "rows_highest": 110,
      "rows_lowest": 301
    },
    {
      "dimension": "platform",
      "metric": "rows_per_user",
      "highest_segment": "unknown",
      "highest_mean": 11.302,
      "lowest_segment": "iOS",
      "lowest_mean": 10.428,
      "ratio": 1.084,
      "rows_highest": 53,
      "rows_lowest": 2027
    },
    {
      "dimension": "content_type",
      "metric": "rows_per_user",
      "highest_segment": "video",
      "highest_mean": 1.984,
      "lowest_segment": "podcast",
      "lowest_mean": 1.902,
      "ratio": 1.043,
      "rows_highest": 2506,
      "rows_lowest": 2545
    }
  ],
  "correlations": [],
  "cohort_compare": {
    "break_week": "2026-06-01",
    "note": "rows_first_14d_per_user рахується лише для користувачів, у яких є повні 14 днів спостереження",
    "before_new_users": 2189,
    "before_rows_first_14d_per_user": 11.71,
    "after_new_users": 1811,
    "after_rows_first_14d_per_user": 5.92,
    "largest_structure_shifts": [
      {
        "dimension": "country",
        "segment": "UA",
        "share_before_pct": 34.2,
        "share_after_pct": 29.1,
        "shift_pp": -5.1
      },
      {
        "dimension": "country",
        "segment": "BR",
        "share_before_pct": 0.0,
        "share_after_pct": 4.1,
        "shift_pp": 4.1
      },
      {
        "dimension": "platform",
        "segment": "iOS",
        "share_before_pct": 49.2,
        "share_after_pct": 52.5,
        "shift_pp": 3.4
      },
      {
        "dimension": "platform",
        "segment": "Android",
        "share_before_pct": 49.5,
        "share_after_pct": 46.2,
        "shift_pp": -3.3
      },
      {
        "dimension": "country",
        "segment": "IN",
        "share_before_pct": 0.0,
        "share_after_pct": 3.1,
        "shift_pp": 3.1
      }
    ]
  }
}
```

## 2. Гіпотези

```json
{
  "hypotheses": [
    {
      "hypothesis": "Приток нових користувачів з Бразилії (BR) та Індії (IN) частково пояснює зростання середньої кількості нових користувачів на тиждень",
      "evidence": "Після 2026-06-01 частка BR = 4.1% (була 0%), частка IN = 3.1% (була 0%); середня кількість нових користувачів на тиждень зросла з 100.5 до 193.2 (ratio 1.923)",
      "how_to_check": "Порівняти динаміку нових користувачів за країнами за тижні до/після break_week; перевірити маркетингові кампанії або партнерства в BR та IN",
      "confidence": "low"
    },
    {
      "hypothesis": "Зменшення ранньої активності нових користувачів (rows_first_14d_per_user з 11.71 до 5.92) пов'язане зі зміною міксу платформ: зростання частки iOS (+3.3 п.п.) та спад Android (-3.3 п.п.)",
      "evidence": "share_before_pct iOS 49.2% -> share_after_pct 52.5% (+3.3 п.п.); Android 49.5% -> 46.2% (-3.3 п.п.); rows_first_14d_per_user впав на 49%",
      "how_to_check": "Розрахувати rows_first_14d_per_user окремо для iOS та Android до та після break_week; перевірити, чи відрізняється залучення на платформах",
      "confidence": "medium"
    },
    {
      "hypothesis": "Нові користувачі з країн, що з'явилися після break_week (BR, IN), мають нижчу ранню активність, що тягне вниз загальний rows_first_14d_per_user",
      "evidence": "BR share_after_pct 4.1%, IN 3.1%; rows_first_14d_per_user після = 5.92 vs до = 11.71",
      "how_to_check": "Сегментувати rows_first_14d_per_user по країнах (UA, BR, IN, інші) для періодів до/після; порівняти середні значення",
      "confidence": "medium"
    },
    {
      "hypothesis": "Зростання частки iOS може бути наслідком оновлення iOS-додатку або ASO-оптимізації, що привабило більше користувачів",
      "evidence": "iOS share_after_pct 52.5% vs 49.2% до (+3.3 п.п.); загальний приріст нових користувачів на тиждень +92%",
      "how_to_check": "Перевірити дати релізів iOS-версії, зміни в App Store (keywords, screenshots) навколо 2026-06-01; порівняти organic vs paid installs для iOS",
      "confidence": "low"
    },
    {
      "hypothesis": "Висока інтенсивність події view_content (rows_per_user 6.45) порівняно з cancel_subscription (1.0) вказує на те, що більшість нових користувачів лише переглядає контент, не конвертюючись",
      "evidence": "event_name view_content highest_mean 6.451 (rows_highest 3830), cancel_subscription lowest_mean 1.0 (rows_lowest 64)",
      "how_to_check": "Обчислити конверсію з view_content у purchase/subscription за когортами до/після break_week; перевірити зміну воронки",
      "confidence": "medium"
    },
    {
      "hypothesis": "Невідома країна (country=unknown) має аномально високий rows_per_user (12.237), що може завадити коректному оцінюванні якості трафіку",
      "evidence": "country unknown highest_mean 12.237 (rows_highest 97) vs IN lowest_mean 5.719 (rows_lowest 57)",
      "how_to_check": "Ідентифікувати джерело трафіку для country=unknown (VPN, proxy, помилка геолокації); вилучити або перерозподілити ці записи і перерахувати метрики",
      "confidence": "low"
    },
    {
      "hypothesis": "Зміна структури каналів (channel unknown зростає) може свідчити про проблеми з атрибуцією джерел трафіку після break_week",
      "evidence": "channel unknown highest_mean 10.945 (rows_highest 110) vs apple_search lowest_mean 10.073 (rows_lowest 301); ratio 1.087",
      "how_to_check": "Перевірити налаштування UTM/атрибуції для періоду після 2026-06-01; порівняти розподіл channel до/після",
      "confidence": "low"
    }
  ],
  "unverified_numbers": [
    "92"
  ]
}
```

## 3. Natural language -> Pandas

Питання: Покажи топ-5 країн за кількістю унікальних користувачів, які купили підписку

Код:
```python
purchase_df = df[df['event_name'] == 'purchase']
result = purchase_df.groupby('country')['user_id'].nunique().sort_values(ascending=False).head(5).reset_index()
result.columns = ['country', 'unique_users']
```

Пояснення: Фільтруємо події покупки, групуємо за країною, рахуємо унікальних користувачів, сортуємо за спаданням і беремо топ-5.

Результат:
```
  country  unique_users
0      UA           114
1      PL            43
2      DE            30
3      US            23
4      GB            22
```

## 3. Natural language -> Pandas

Питання: Скільки в середньому подій view_content на користувача в кожному місяці?

Код:
```python
view_df = df[df['event_name'] == 'view_content'].copy()
view_df['month'] = pd.to_datetime(view_df['event_date']).dt.to_period('M')
user_month_counts = view_df.groupby(['month', 'user_id']).size().reset_index(name='count')
result = user_month_counts.groupby('month')['count'].mean()
```

Пояснення: Фільтрує події view_content, створює колонку місяця, рахує кількість подій на кожного користувача в кожному місяці, потім обчислює середнє значення цих кількостей по місяцях.

Результат:
```
month
2026-01    6.923423
2026-02    5.899306
2026-03    6.035714
2026-04    5.952744
2026-05    5.751987
2026-06    3.677487
2026-07    3.155767
2026-08    2.625668
Freq: M
```

## 3. Natural language -> Pandas

Питання: Яка погода завтра у Києві?

Код:
```python
(не згенеровано)
```

Пояснення: 

Помилка: Модель не повернула код (можливо, питання не стосується даних): Модель не повернула коректний JSON: немає полів: ['code', 'explanation']

## 4. Пояснення зміни метрики: розклад по сегментах (порахував код)

```json
{
  "metric": "count(*) where event_name == 'view_content'",
  "period_before": [
    "2026-03-01",
    "2026-05-31"
  ],
  "period_after": [
    "2026-06-01",
    "2026-08-05"
  ],
  "per_day_before": 125.859,
  "per_day_after": 100.864,
  "relative_change_pct": -19.9,
  "segments": {
    "channel": [
      {
        "segment": "google_ads",
        "rows_before": 2579,
        "rows_after": 1353,
        "low_volume": false,
        "per_day_before": 28.033,
        "per_day_after": 20.5,
        "change": -7.533,
        "relative_change_pct": -26.9,
        "share_of_total_change_pct": 30.1
      },
      {
        "segment": "organic",
        "rows_before": 3368,
        "rows_after": 1945,
        "low_volume": false,
        "per_day_before": 36.609,
        "per_day_after": 29.47,
        "change": -7.139,
        "relative_change_pct": -19.5,
        "share_of_total_change_pct": 28.6
      },
      {
        "segment": "facebook_ads",
        "rows_before": 2169,
        "rows_after": 1251,
        "low_volume": false,
        "per_day_before": 23.576,
        "per_day_after": 18.955,
        "change": -4.622,
        "relative_change_pct": -19.6,
        "share_of_total_change_pct": 18.5
      },
      {
        "segment": "email",
        "rows_before": 1021,
        "rows_after": 529,
        "low_volume": false,
        "per_day_before": 11.098,
        "per_day_after": 8.015,
        "change": -3.083,
        "relative_change_pct": -27.8,
        "share_of_total_change_pct": 12.3
      },
      {
        "segment": "unknown",
        "rows_before": 375,
        "rows_after": 161,
        "low_volume": false,
        "per_day_before": 4.076,
        "per_day_after": 2.439,
        "change": -1.637,
        "relative_change_pct": -40.2,
        "share_of_total_change_pct": 6.5
      },
      {
        "segment": "referral",
        "rows_before": 1366,
        "rows_after": 912,
        "low_volume": false,
        "per_day_before": 14.848,
        "per_day_after": 13.818,
        "change": -1.03,
        "relative_change_pct": -6.9,
        "share_of_total_change_pct": 4.1
      },
      {
        "segment": "apple_search",
        "rows_before": 701,
        "rows_after": 506,
        "low_volume": false,
        "per_day_before": 7.62,
        "per_day_after": 7.667,
        "change": 0.047,
        "relative_change_pct": 0.6,
        "share_of_total_change_pct": -0.2
      }
    ],
    "platform": [
      {
        "segment": "Android",
        "rows_before": 5621,
        "rows_after": 3056,
        "low_volume": false,
        "per_day_before": 61.098,
        "per_day_after": 46.303,
        "change": -14.795,
        "relative_change_pct": -24.2,
        "share_of_total_change_pct": 59.2
      },
      {
        "segment": "iOS",
        "rows_before": 5838,
        "rows_after": 3511,
        "low_volume": false,
        "per_day_before": 63.457,
        "per_day_after": 53.197,
        "change": -10.26,
        "relative_change_pct": -16.2,
        "share_of_total_change_pct": 41.0
      },
      {
        "segment": "unknown",
        "rows_before": 120,
        "rows_after": 90,
        "low_volume": false,
        "per_day_before": 1.304,
        "per_day_after": 1.364,
        "change": 0.059,
        "relative_change_pct": 4.5,
        "share_of_total_change_pct": -0.2
      }
    ],
    "country": [
      {
        "segment": "UA",
        "rows_before": 3791,
        "rows_after": 1935,
        "low_volume": false,
        "per_day_before": 41.207,
        "per_day_after": 29.318,
        "change": -11.888,
        "relative_change_pct": -28.9,
        "share_of_total_change_pct": 47.6
      },
      {
        "segment": "PL",
        "rows_before": 1594,
        "rows_after": 724,
        "low_volume": false,
        "per_day_before": 17.326,
        "per_day_after": 10.97,
        "change": -6.356,
        "relative_change_pct": -36.7,
        "share_of_total_change_pct": 25.4
      },
      {
        "segment": "BR",
        "rows_before": 0,
        "rows_after": 260,
        "low_volume": true,
        "per_day_before": 0.0,
        "per_day_after": 3.939,
        "change": 3.939,
        "relative_change_pct": null,
        "share_of_total_change_pct": -15.8
      },
      {
        "segment": "US",
        "rows_before": 1155,
        "rows_after": 595,
        "low_volume": false,
        "per_day_before": 12.554,
        "per_day_after": 9.015,
        "change": -3.539,
        "relative_change_pct": -28.2,
        "share_of_total_change_pct": 14.2
      },
      {
        "segment": "FR",
        "rows_before": 881,
        "rows_after": 414,
        "low_volume": false,
        "per_day_before": 9.576,
        "per_day_after": 6.273,
        "change": -3.303,
        "relative_change_pct": -34.5,
        "share_of_total_change_pct": 13.2
      },
      {
        "segment": "IN",
        "rows_before": 0,
        "rows_after": 170,
        "low_volume": true,
        "per_day_before": 0.0,
        "per_day_after": 2.576,
        "change": 2.576,
        "relative_change_pct": null,
        "share_of_total_change_pct": -10.3
      },
      {
        "segment": "GB",
        "rows_before": 738,
        "rows_after": 425,
        "low_volume": false,
        "per_day_before": 8.022,
        "per_day_after": 6.439,
        "change": -1.582,
        "relative_change_pct": -19.7,
        "share_of_total_change_pct": 6.3
      },
      {
        "segment": "DE",
        "rows_before": 1118,
        "rows_after": 716,
        "low_volume": false,
        "per_day_before": 12.152,
        "per_day_after": 10.848,
        "change": -1.304,
        "relative_change_pct": -10.7,
        "share_of_total_change_pct": 5.2
      },
      {
        "segment": "unknown",
        "rows_before": 354,
        "rows_after": 182,
        "low_volume": false,
        "per_day_before": 3.848,
        "per_day_after": 2.758,
        "change": -1.09,
        "relative_change_pct": -28.3,
        "share_of_total_change_pct": 4.4
      },
      {
        "segment": "IT",
        "rows_before": 593,
        "rows_after": 376,
        "low_volume": false,
        "per_day_before": 6.446,
        "per_day_after": 5.697,
        "change": -0.749,
        "relative_change_pct": -11.6,
        "share_of_total_change_pct": 3.0
      },
      {
        "segment": "CA",
        "rows_before": 404,
        "rows_after": 241,
        "low_volume": false,
        "per_day_before": 4.391,
        "per_day_after": 3.652,
        "change": -0.74,
        "relative_change_pct": -16.8,
        "share_of_total_change_pct": 3.0
      },
      {
        "segment": "ES",
        "rows_before": 581,
        "rows_after": 376,
        "low_volume": false,
        "per_day_before": 6.315,
        "per_day_after": 5.697,
        "change": -0.618,
        "relative_change_pct": -9.8,
        "share_of_total_change_pct": 2.5
      },
      {
        "segment": "NL",
        "rows_before": 370,
        "rows_after": 243,
        "low_volume": false,
        "per_day_before": 4.022,
        "per_day_after": 3.682,
        "change": -0.34,
        "relative_change_pct": -8.5,
        "share_of_total_change_pct": 1.4
      }
    ]
  },
  "concentration": {
    "channel": {
      "verdict": "concentrated_or_mixed",
      "mean_relative_change_pct": -20.0,
      "std_relative_change_pct": 12.6,
      "largest_segment": "google_ads"
    },
    "platform": {
      "verdict": "concentrated_or_mixed",
      "mean_relative_change_pct": -12.0,
      "std_relative_change_pct": 12.1,
      "largest_segment": "Android"
    },
    "country": {
      "verdict": "concentrated_or_mixed",
      "mean_relative_change_pct": -21.2,
      "std_relative_change_pct": 10.0,
      "largest_segment": "UA"
    }
  }
}
```

## 4. Пояснення зміни метрики: висновок LLM

```json
{
  "summary": "Загальний показник view_content впав на 19,9% (з 125,9 до 100,9 подій на день). Зміна зосереджена в кількох ключових сегментах: країни UA та PL (разом ~73% зміни), платформа Android (~59%), канали google_ads та organic (разом ~59%). Інші сегменти дають менший внесок.",
  "main_drivers": [
    "UA (country)",
    "PL (country)",
    "Android (platform)",
    "google_ads (channel)",
    "organic (channel)"
  ],
  "hypothesis": "Спад викликаний зменшенням трафіку з Android-пристроїв в Україні та Польщі через платні (google_ads) та органічні канали, можливо через скорочення рекламного бюджету або зміни в алгоритмах пошуку.",
  "checks": [
    "Перевірити витрати на google_ads та facebook_ads для UA/PL у період after",
    "Перевірити наявність змін у трекінгу подій на Android (наприклад, оновлення SDK)",
    "Проаналізувати показники органічного пошуку (impressions, CTR) для основних ключових слів в UA/PL"
  ],
  "unverified_numbers": [
    "73"
  ]
}
```

## 5. Executive summary

```json
{
  "what_happened": "Після 1 червня 2026 р. перегляди контенту (view_content) впали на 20 % (з 125,9 до 100,9 на день), кількість нових користувачів на тиждень зросла майже вдвічі, але середня активність нових користувачів за перші 14 днів зменшилася з 11,7 до 5,9 дій. Змінився склад аудиторії: частка UA впала на 5,1 п.п., з’явилися BR (4,1 п.п.) та IN (3,1 п.п.), iOS виросла на 3,4 п.п., Android впала на 3,3 п.п.",
  "why_it_matters": "Падіння переглядів і активності загрожує утриманню та виручці, навіть при рості нових користувачів. Зміна географії та платформи може вплинути на монетизацію та продуктивність маркетингу.",
  "next_steps": [
    "Перевірити зміни в продукті/техніці навколо 1 червня, які могли зменшити view_content.",
    "Проаналізувати воронку нових когорт (BR, IN, iOS) для виявлення причин низької активності.",
    "Оцінити вплив зміни гео/платформи на ARPU та скоррегувати стратегію набору та онбордингу."
  ],
  "unverified_numbers": []
}
```