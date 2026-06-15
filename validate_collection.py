"""
validate_collection.py
Comprehensive data quality report for ustech_ohlcv / ustech_verified.

Checks:
  1. Row counts per timeframe (raw vs verified)
  2. Year-by-year candle breakdown
  3. Gap detection  — weekdays with < 50% expected candles
  4. Calendar gaps  — whole weeks/months missing
  5. Duplicate detection in the raw table (via DB UNIQUE constraint violations)
  6. OHLCV logic integrity  (H >= O/C/L, L <= O/C, no zero prices)
  7. Price-range sanity for USTECm (10,000 – 25,000 typical range)
  8. Volume presence

Usage:
    python validate_collection.py
    python validate_collection.py --start 2023-01-01 --end 2026-05-31
    python validate_collection.py --timeframe 5min
"""

import sys
import os
import argparse
from datetime import date, datetime, timedelta
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import psycopg2
from config import DB_CONFIG, TIMEFRAMES, SYMBOL

EXPECTED_CANDLES_PER_DAY = {
    "5min":  78,
    "15min": 26,
    "1H":    7,
    "4H":    2,
}

USTECH_PRICE_MIN = 8_000.0
USTECH_PRICE_MAX = 30_000.0

LINE = "=" * 62


def get_conn():
    return psycopg2.connect(**DB_CONFIG)


def _q(conn, sql, params=()):
    with conn.cursor() as cur:
        cur.execute(sql, params)
        return cur.fetchall()


# ─────────────────────────────────────────────────────────────────────────────
# 1. Row counts
# ─────────────────────────────────────────────────────────────────────────────
def report_counts(conn, symbol, start_date, end_date):
    print(f"\n{LINE}")
    print("  1. ROW COUNTS (raw  vs  verified)")
    print(LINE)

    for table in ("ustech_ohlcv", "ustech_verified"):
        try:
            rows = _q(conn, f"""
                SELECT timeframe, COUNT(*) FROM {table}
                WHERE symbol = %s
                  AND timestamp::date BETWEEN %s AND %s
                GROUP BY timeframe ORDER BY timeframe
            """, (symbol, start_date, end_date))
            print(f"\n  [{table}]")
            if not rows:
                print("    (no data)")
            else:
                for tf, cnt in rows:
                    print(f"    {tf:<8s}: {cnt:>12,} candles")
        except psycopg2.errors.UndefinedTable:
            conn.rollback()
            print(f"    (table {table!r} does not exist yet)")


# ─────────────────────────────────────────────────────────────────────────────
# 2. Year-by-year breakdown
# ─────────────────────────────────────────────────────────────────────────────
def report_yearly(conn, symbol, start_date, end_date):
    print(f"\n{LINE}")
    print("  2. YEAR-BY-YEAR CANDLE COUNTS  (raw table)")
    print(LINE)

    rows = _q(conn, """
        SELECT timeframe, EXTRACT(YEAR FROM timestamp)::int AS yr, COUNT(*)
        FROM ustech_ohlcv
        WHERE symbol = %s
          AND timestamp::date BETWEEN %s AND %s
        GROUP BY timeframe, yr
        ORDER BY timeframe, yr
    """, (symbol, start_date, end_date))

    if not rows:
        print("  (no data)")
        return

    # pivot: {tf: {year: count}}
    data = defaultdict(dict)
    years_seen = set()
    for tf, yr, cnt in rows:
        data[tf][yr] = cnt
        years_seen.add(yr)

    years = sorted(years_seen)
    header = f"  {'TF':<8}" + "".join(f"  {y:>10}" for y in years) + f"  {'TOTAL':>10}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    tf_order = {tf: i for i, (tf, _) in enumerate(TIMEFRAMES)}
    for tf in sorted(data, key=lambda t: tf_order.get(t, 99)):
        row_total = sum(data[tf].values())
        line = f"  {tf:<8}"
        for yr in years:
            line += f"  {data[tf].get(yr, 0):>10,}"
        line += f"  {row_total:>10,}"
        print(line)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Gap detection
# ─────────────────────────────────────────────────────────────────────────────
def report_gaps(conn, symbol, start_date, end_date, threshold=0.5):
    print(f"\n{LINE}")
    print(f"  3. GAP DETECTION  (weekdays with < {threshold*100:.0f}% expected candles)")
    print(LINE)

    total_gap_days = 0
    for tf, _ in TIMEFRAMES:
        expected = EXPECTED_CANDLES_PER_DAY.get(tf, 0)
        if not expected:
            continue

        rows = _q(conn, """
            SELECT timestamp::date AS day, COUNT(*) AS cnt
            FROM ustech_ohlcv
            WHERE timeframe = %s AND symbol = %s
              AND timestamp::date BETWEEN %s AND %s
              AND EXTRACT(DOW FROM timestamp) NOT IN (0, 6)
            GROUP BY day
            ORDER BY day
        """, (tf, symbol, start_date, end_date))

        gap_rows = [(d, int(c), expected) for d, c in rows if c < expected * threshold]
        total_gap_days += len(gap_rows)

        if gap_rows:
            print(f"\n  {tf}: {len(gap_rows)} gap days")
            for day, actual, exp in gap_rows[:20]:
                pct = actual / exp * 100
                print(f"    {day}  {actual:>4}/{exp}  ({pct:.0f}%)")
            if len(gap_rows) > 20:
                print(f"    ... and {len(gap_rows)-20} more")
        else:
            print(f"\n  {tf}: no gap days  [OK]")

    print(f"\n  Total gap-days across all timeframes: {total_gap_days}")


# ─────────────────────────────────────────────────────────────────────────────
# 4. Calendar coverage — missing months
# ─────────────────────────────────────────────────────────────────────────────
def report_calendar_coverage(conn, symbol, start_date, end_date):
    print(f"\n{LINE}")
    print("  4. CALENDAR COVERAGE  (months present in raw table, per timeframe)")
    print(LINE)

    tf_months = {}
    for tf, _ in TIMEFRAMES:
        rows = _q(conn, """
            SELECT DISTINCT TO_CHAR(timestamp, 'YYYY-MM') AS ym
            FROM ustech_ohlcv
            WHERE timeframe = %s AND symbol = %s
              AND timestamp::date BETWEEN %s AND %s
            ORDER BY ym
        """, (tf, symbol, start_date, end_date))
        tf_months[tf] = {r[0] for r in rows}

    # Build expected month list
    d = start_date.replace(day=1)
    expected_months = []
    while d <= end_date:
        expected_months.append(d.strftime("%Y-%m"))
        # advance one month
        if d.month == 12:
            d = d.replace(year=d.year + 1, month=1)
        else:
            d = d.replace(month=d.month + 1)

    for tf, _ in TIMEFRAMES:
        present = tf_months.get(tf, set())
        missing = [m for m in expected_months if m not in present]
        if missing:
            print(f"\n  {tf}: missing {len(missing)} month(s)")
            for m in missing:
                print(f"    {m}")
        else:
            print(f"  {tf}: all {len(expected_months)} months present  [OK]")


# ─────────────────────────────────────────────────────────────────────────────
# 5. Duplicate detection (DB level)
# ─────────────────────────────────────────────────────────────────────────────
def report_duplicates(conn, symbol, start_date, end_date):
    print(f"\n{LINE}")
    print("  5. DUPLICATE DETECTION  (same symbol + timeframe + timestamp)")
    print(LINE)

    rows = _q(conn, """
        SELECT timeframe, COUNT(*) AS total_rows,
               COUNT(DISTINCT timestamp)  AS distinct_ts,
               COUNT(*) - COUNT(DISTINCT timestamp) AS dup_count
        FROM ustech_ohlcv
        WHERE symbol = %s
          AND timestamp::date BETWEEN %s AND %s
        GROUP BY timeframe
        ORDER BY timeframe
    """, (symbol, start_date, end_date))

    any_dups = False
    for tf, total, distinct, dups in rows:
        status = "[OK]" if dups == 0 else f"*** {dups:,} DUPLICATES ***"
        print(f"  {tf:<8s}: {total:>10,} rows | {distinct:>10,} unique timestamps | {status}")
        if dups > 0:
            any_dups = True

    if not any_dups:
        print("\n  No duplicates found across any timeframe.")


# ─────────────────────────────────────────────────────────────────────────────
# 6. OHLCV logic integrity
# ─────────────────────────────────────────────────────────────────────────────
def report_ohlcv_integrity(conn, symbol, start_date, end_date):
    print(f"\n{LINE}")
    print("  6. OHLCV LOGIC INTEGRITY")
    print(LINE)

    checks = [
        ("high < low",          "high < low"),
        ("high < open",         "high < open"),
        ("high < close",        "high < close"),
        ("low  > open",         "low  > open"),
        ("low  > close",        "low  > close"),
        ("zero/negative open",  "open <= 0"),
        ("zero/negative high",  "high <= 0"),
        ("zero/negative low",   "low  <= 0"),
        ("zero/negative close", "close <= 0"),
        ("zero volume",         "volume = 0"),
    ]

    any_issues = False
    for label, condition in checks:
        rows = _q(conn, f"""
            SELECT COUNT(*) FROM ustech_ohlcv
            WHERE symbol = %s
              AND timestamp::date BETWEEN %s AND %s
              AND {condition}
        """, (symbol, start_date, end_date))
        count = rows[0][0] if rows else 0
        status = "[OK]" if count == 0 else f"*** {count:,} VIOLATIONS ***"
        print(f"  {label:<25s}: {status}")
        if count > 0:
            any_issues = True

    if not any_issues:
        print("\n  All OHLCV logic checks passed.")


# ─────────────────────────────────────────────────────────────────────────────
# 7. Price range sanity
# ─────────────────────────────────────────────────────────────────────────────
def report_price_sanity(conn, symbol, start_date, end_date):
    print(f"\n{LINE}")
    print(f"  7. PRICE RANGE SANITY  ({USTECH_PRICE_MIN:,.0f} – {USTECH_PRICE_MAX:,.0f}  for USTECm)")
    print(LINE)

    rows = _q(conn, """
        SELECT timeframe,
               MIN(low)    AS min_price,
               MAX(high)   AS max_price,
               AVG(close)  AS avg_close,
               COUNT(*)    FILTER (WHERE low  < %s) AS below_min,
               COUNT(*)    FILTER (WHERE high > %s) AS above_max
        FROM ustech_ohlcv
        WHERE symbol = %s
          AND timestamp::date BETWEEN %s AND %s
        GROUP BY timeframe
        ORDER BY timeframe
    """, (USTECH_PRICE_MIN, USTECH_PRICE_MAX, symbol, start_date, end_date))

    any_issues = False
    for tf, mn, mx, avg, below, above in rows:
        issues = []
        if below and below > 0:
            issues.append(f"{below} below min")
        if above and above > 0:
            issues.append(f"{above} above max")
        status = "  [" + ", ".join(issues) + "]" if issues else "  [OK]"
        print(f"  {tf:<8s}: min={float(mn):>10,.2f}  max={float(mx):>10,.2f}  avg={float(avg):>10,.2f}{status}")
        if issues:
            any_issues = True

    if not any_issues:
        print("\n  All prices within expected USTECm range.")


# ─────────────────────────────────────────────────────────────────────────────
# 8. Volume presence
# ─────────────────────────────────────────────────────────────────────────────
def report_volume(conn, symbol, start_date, end_date):
    print(f"\n{LINE}")
    print("  8. VOLUME PRESENCE")
    print(LINE)

    rows = _q(conn, """
        SELECT timeframe,
               COUNT(*) FILTER (WHERE volume = 0)  AS zero_vol,
               COUNT(*) FILTER (WHERE volume IS NULL) AS null_vol,
               AVG(volume) AS avg_vol,
               MAX(volume) AS max_vol
        FROM ustech_ohlcv
        WHERE symbol = %s
          AND timestamp::date BETWEEN %s AND %s
        GROUP BY timeframe
        ORDER BY timeframe
    """, (symbol, start_date, end_date))

    for tf, zero, null, avg, mx in rows:
        issues = []
        if zero > 0:  issues.append(f"{zero:,} zero-volume candles")
        if null > 0:  issues.append(f"{null:,} null-volume candles")
        status = "  [" + ", ".join(issues) + "]" if issues else "  [OK]"
        print(f"  {tf:<8s}: avg={float(avg):>10,.1f}  max={float(mx):>12,.1f}{status}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="MT5 data quality report")
    parser.add_argument("--start",     default="2023-01-01")
    parser.add_argument("--end",       default="2026-05-31")
    parser.add_argument("--symbol",    default=SYMBOL)
    parser.add_argument("--timeframe", default=None, help="Filter to one timeframe")
    args = parser.parse_args()

    start_date = date.fromisoformat(args.start)
    end_date   = date.fromisoformat(args.end)
    symbol     = args.symbol

    print(f"\n{LINE}")
    print("  MT5 COLLECTION  —  DATA QUALITY REPORT")
    print(LINE)
    print(f"  Symbol : {symbol}")
    print(f"  Range  : {start_date}  to  {end_date}")
    print(f"  Run at : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    try:
        conn = get_conn()
    except Exception as e:
        print(f"\n  ERROR: cannot connect to database — {e}")
        sys.exit(1)

    try:
        report_counts(conn, symbol, start_date, end_date)
        report_yearly(conn, symbol, start_date, end_date)
        report_gaps(conn, symbol, start_date, end_date)
        report_calendar_coverage(conn, symbol, start_date, end_date)
        report_duplicates(conn, symbol, start_date, end_date)
        report_ohlcv_integrity(conn, symbol, start_date, end_date)
        report_price_sanity(conn, symbol, start_date, end_date)
        report_volume(conn, symbol, start_date, end_date)
    finally:
        conn.close()

    print(f"\n{LINE}")
    print("  REPORT COMPLETE")
    print(LINE)


if __name__ == "__main__":
    main()
