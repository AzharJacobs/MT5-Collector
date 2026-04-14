"""
Build ML Datasets — Zone-to-Zone Strategy
One-click script to generate ML-ready datasets from your collected OHLCV data.

Run AFTER mt5_collector.py has populated the database.
"""

import os
import sys

def main():
    print("=" * 60)
    print("Zone-to-Zone ML Dataset Builder")
    print("=" * 60)
    print()
    print("This will:")
    print("  1. Load OHLCV data from PostgreSQL")
    print("  2. Detect supply & demand zones")
    print("  3. Add confirmation signals (engulfing, pin bars, BOS)")
    print("  4. Add technical indicators (ATR, RSI, EMA, BB)")
    print("  5. Inject 1H + 4H HTF context onto LTF rows")
    print("  6. Generate trade labels (Zone-to-Zone rules)")
    print("  7. Export train/test CSV splits to ./datasets/")
    print()

    try:
        from ml_dataset import build_ml_dataset, export_train_test_split, print_dataset_summary

        # Target timeframes for scalping + intraday
        timeframes = ["5min", "15min", "1H"]

        os.makedirs("./datasets", exist_ok=True)

        all_results = {}
        for tf in timeframes:
            print(f"\n--- Processing {tf} ---")
            df = build_ml_dataset(
                timeframe=tf,
                symbol="USTECm",
                max_label_bars=50,
                min_rr=1.5,
                require_confirmation=True,
                use_midline_tp=True,
            )

            if df.empty:
                print(f"  No data for {tf}. Make sure mt5_collector.py has run first.")
                continue

            all_results[tf] = df
            print_dataset_summary(df, tf)

            # Time-based train/test split (80/20)
            export_train_test_split(df, "./datasets", tf, test_ratio=0.2)

        print("\n" + "=" * 60)
        print("DONE. Files saved to ./datasets/")
        print("=" * 60)
        for tf in all_results:
            print(f"  ./datasets/{tf}_train.csv")
            print(f"  ./datasets/{tf}_test.csv")
        print()
        print("Next step: train your XGBoost/RandomForest classifier")
        print("  Features: all columns NOT in [timestamp, symbol, timeframe,")
        print("            signal, signal_reason, trade_outcome, label,")
        print("            tp_price, sl_price, rr_ratio, open, high, low, close]")
        print("  Target:   'label' column (1=buy, -1=sell, 0=hold)")

    except ImportError as e:
        print(f"Import error: {e}")
        print("Make sure all dependencies are installed:")
        print("  pip install -r requirements.txt")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()