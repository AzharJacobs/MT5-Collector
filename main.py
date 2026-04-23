"""
main.py — Full Pipeline Entry Point
Runs setup checks then executes the complete data collection pipeline.
"""

import os
import sys
import shutil


def check_env_file() -> bool:
    if not os.path.exists('.env'):
        if os.path.exists('.env.example'):
            print("Creating .env from template...")
            shutil.copy('.env.example', '.env')
            print("=" * 60)
            print("IMPORTANT: Edit .env with your credentials:")
            print("  DB_USER, DB_PASSWORD — PostgreSQL")
            print("  MT5_LOGIN, MT5_PASSWORD, MT5_SERVER — MetaTrader 5")
            print("  SYMBOL — trading symbol (default: USTECm)")
            print("=" * 60)
            print("\nAfter editing .env, run main.py again.")
            return False
        print("Error: .env.example not found.")
        return False
    return True


def check_dependencies() -> bool:
    required = ['MetaTrader5', 'psycopg2', 'dotenv', 'pandas', 'numpy']
    missing = []
    for pkg in required:
        try:
            __import__('dotenv' if pkg == 'dotenv' else pkg)
        except ImportError:
            missing.append(pkg)

    if missing:
        print(f"Missing packages: {missing}. Installing from requirements.txt...")
        os.system(f"{sys.executable} -m pip install -r requirements.txt")
    return True


def main():
    print("=" * 60)
    print("MT5 OHLCV Data Collector — Full Pipeline")
    print("=" * 60)

    if not check_env_file():
        sys.exit(1)

    check_dependencies()

    try:
        from collectors.price_collector import MT5Collector

        print("\nStarting data collection...")
        print("-" * 60)

        collector = MT5Collector()
        results = collector.run(setup_db=True, incremental=True)

        print("\n" + "=" * 60)
        if results['success']:
            print("SUCCESS! Data collection completed.")
            if 'summary' in results:
                print("\nDatabase now contains:")
                total = 0
                for row in results['summary']:
                    count = row['total_candles']
                    total += count
                    print(f"  {row['timeframe']:8s}: {count:,} candles")
                print(f"\n  Total: {total:,} candles")
        else:
            print(f"FAILED: {results['message']}")
        print("=" * 60)

    except ImportError as e:
        print(f"\nImport error: {e}")
        print("Run: pip install -r requirements.txt")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
