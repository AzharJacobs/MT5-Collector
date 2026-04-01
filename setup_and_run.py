"""
Setup and Run Script
One-click script to setup database and start data collection
"""

import os
import sys
import shutil

def check_env_file():
    """Check if .env file exists, create from template if not"""
    if not os.path.exists('.env'):
        if os.path.exists('.env.example'):
            print("Creating .env file from template...")
            shutil.copy('.env.example', '.env')
            print("=" * 60)
            print("IMPORTANT: Please edit the .env file with your settings:")
            print("  - DB_USER: Your PostgreSQL username")
            print("  - DB_PASSWORD: Your PostgreSQL password")
            print("  - MT5_LOGIN (optional): Your MT5 account login")
            print("  - SYMBOL: The trading symbol (default: USTech)")
            print("=" * 60)
            print("\nAfter editing .env, run this script again.")
            return False
        else:
            print("Error: .env.example file not found!")
            return False
    return True

def check_dependencies():
    """Check if required packages are installed"""
    required = ['MetaTrader5', 'psycopg2', 'dotenv', 'pandas', 'numpy']
    missing = []

    for package in required:
        try:
            if package == 'dotenv':
                __import__('dotenv')
            elif package == 'psycopg2':
                __import__('psycopg2')
            else:
                __import__(package)
        except ImportError:
            missing.append(package)

    if missing:
        print("Missing packages detected. Installing...")
        os.system(f"{sys.executable} -m pip install -r requirements.txt")
        return False
    return True

def main():
    print("=" * 60)
    print("MT5 OHLCV Data Collector - Setup & Run")
    print("=" * 60)

    # Check environment file
    if not check_env_file():
        sys.exit(1)

    # Check dependencies
    check_dependencies()

    # Import and run collector
    try:
        from mt5_collector import MT5Collector

        print("\nStarting data collection...")
        print("-" * 60)

        collector = MT5Collector()
        results = collector.run(setup_db=True, incremental=True)

        if results['success']:
            print("\n" + "=" * 60)
            print("SUCCESS! Data collection completed.")
            print("=" * 60)

            if 'summary' in results:
                print("\nDatabase now contains:")
                total = 0
                for row in results['summary']:
                    count = row['total_candles']
                    total += count
                    print(f"  {row['timeframe']:8s}: {count:,} candles")
                print(f"\n  Total: {total:,} candles")
        else:
            print("\n" + "=" * 60)
            print(f"FAILED: {results['message']}")
            print("=" * 60)

    except ImportError as e:
        print(f"\nError importing modules: {e}")
        print("Make sure all dependencies are installed:")
        print("  pip install -r requirements.txt")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
