# Feature set definitions used in the experiments.

FEATURE_SETS = {
    # F1: Minimal (price only)
    'F1_price': {
        'features': ['close'],
    },
    # F2: Price + momentum
    'F2_price_momentum': {
        'features': ['close', 'rsi_14', 'macd_hist'],
    },
    # F3: Price + momentum + volatility
    'F3_price_momentum_vol': {
        'features': ['close', 'rsi_14', 'macd_hist', 'bb_pos_20', 'vol_ratio_20'],
    },
    # F4: Price + momentum + volatility + sentiment
    'F4_price_momentum_vol_sentiment': {
        'features': [
            'close',
            'rsi_14',
            'macd_hist',
            'bb_pos_20',
            'vol_ratio_20',
            'fear_greed_index',
            'btc_dominance',
        ],
    },
    # F5: All combined
    'F5_all': {
        'features': [
            'close',
            'rsi_14',
            'macd_hist',
            'bb_pos_20',
            'vol_ratio_20',
            'fear_greed_index',
            'btc_dominance',
            'trend_bitcoin',
            'ret_24',
        ],
    },
}

TRAINING_CONFIGS = {
    'fees': [0.0, 0.001, 0.003],
    'inactivity_penalty': [False, True],
    'n_episodes': 500,
    'alpha': 0.1,
    'gamma': 0.99,
    'epsilon_start': 1.0,
    'epsilon_decay': 0.9995,
    'epsilon_min': 0.01,
}
