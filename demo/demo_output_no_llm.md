# Демо AI-асистента аналітика

Датасет: `demo/events_flat.csv`, 41,997 рядків (події застосунку з лаби №2). Режим: **тільки розрахунки (без LLM)**.


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

## 2. Факти для гіпотез (порахував код)

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