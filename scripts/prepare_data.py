# Prepare the dataset and create chronological train/test splits.

from pathlib import Path

import pandas as pd


def prepare_data(
    raw_data_path: str = 'data/btc_pulse_features.csv',
    train_output_path: str = 'data/splits/btc_train.csv',
    test_output_path: str = 'data/splits/btc_test.csv',
    train_split: float = 0.8,
):
    print('=' * 60)
    print('DATA PREPARATION')
    print('=' * 60)

    df = pd.read_csv(raw_data_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp', ascending=True).reset_index(drop=True)

    key_features = [
        'close',
        'volume',
        'ret_1',
        'ret_24',
        'rsi_14',
        'macd',
        'macd_signal',
        'macd_hist',
        'bb_ma_20',
        'bb_upper_20',
        'bb_lower_20',
        'bb_pos_20',
        'vol_ratio_20',
        'fear_greed_index',
        'btc_dominance',
        'trend_bitcoin',
    ]

    existing_features = [col for col in key_features if col in df.columns]
    df_clean = df.dropna(subset=existing_features)

    split_idx = int(len(df_clean) * train_split)
    train_df = df_clean.iloc[:split_idx].copy()
    test_df = df_clean.iloc[split_idx:].copy()

    Path(Path(train_output_path).parent).mkdir(parents=True, exist_ok=True)
    train_df.to_csv(train_output_path, index=False)
    test_df.to_csv(test_output_path, index=False)

    print(f'Train rows: {len(train_df)}')
    print(f'Test rows:  {len(test_df)}')
    print(f'Train range: {train_df["timestamp"].min()} -> {train_df["timestamp"].max()}')
    print(f'Test range:  {test_df["timestamp"].min()} -> {test_df["timestamp"].max()}')

    return train_df, test_df


if __name__ == '__main__':
    prepare_data()
