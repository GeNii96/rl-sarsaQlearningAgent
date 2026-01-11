# Feature-based trading environment for tabular RL.

import json
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


class FeatureTradingEnv:
    ACTION_HOLD = 0
    ACTION_LONG = 1
    ACTION_SHORT = 2
    ACTION_NAMES = {0: 'HOLD', 1: 'LONG', 2: 'SHORT'}

    def __init__(
        self,
        data_path: str,
        feature_cols: List[str],
        feature_bins: Dict[str, np.ndarray],
        fee: float = 0.0,
        initial_balance: float = 10_000.0,
        inactivity_penalty: float = 0.0,
        verbose: bool = False,
    ):
        self.df = pd.read_csv(data_path)
        self.df['timestamp'] = pd.to_datetime(self.df['timestamp'])
        self.df = self.df.sort_values('timestamp').reset_index(drop=True)

        self.feature_cols = feature_cols
        self.feature_bins = feature_bins
        self.fee = fee
        self.initial_balance = initial_balance
        self.inactivity_penalty = inactivity_penalty
        self.verbose = verbose

        self.state_map = {}
        self.state_id_counter = 0

        self.current_step = 0
        self.max_steps = len(self.df) - 1

        self.initial_balance_usd = initial_balance
        self.balance_usd = initial_balance
        self.position = 0.0
        self.entry_price = 0.0

        self.prev_portfolio_value = initial_balance
        self.portfolio_values = [initial_balance]
        self.trades = []
        self.episode_rewards = []

        self._episode_started = False

    def reset(self) -> int:
        self.current_step = 0
        self.balance_usd = self.initial_balance_usd
        self.position = 0.0
        self.entry_price = 0.0
        self.prev_portfolio_value = self.initial_balance_usd
        self.portfolio_values = [self.initial_balance_usd]
        self.trades = []
        self.episode_rewards = []
        self._episode_started = True
        return self._get_state()

    def step(self, action: int) -> Tuple[int, float, bool, dict]:
        if not self._episode_started:
            raise RuntimeError('Call reset() before step().')

        if self.current_step >= self.max_steps:
            return self._get_state(), 0.0, True, {}

        current_price = self.df.loc[self.current_step, 'close']
        next_price = self.df.loc[self.current_step + 1, 'close']

        portfolio_before = self._calculate_portfolio_value(current_price)
        self._execute_action(action, current_price)
        portfolio_after = self._calculate_portfolio_value(next_price)

        reward = self._calculate_reward(
            portfolio_before=portfolio_before,
            portfolio_after=portfolio_after,
            action=action,
        )

        self.current_step += 1
        next_state = self._get_state()
        done = self.current_step >= self.max_steps

        info = {
            'portfolio_value': portfolio_after,
            'position': self.position,
            'balance': self.balance_usd,
            'current_price': next_price,
            'timestamp': self.df.loc[self.current_step, 'timestamp'],
        }

        self.prev_portfolio_value = portfolio_after
        self.portfolio_values.append(portfolio_after)
        self.episode_rewards.append(reward)

        if self.verbose and self.current_step % 100 == 0:
            print(
                f"Step {self.current_step:5d} | Action: {self.ACTION_NAMES[action]:5s} | "
                f"Reward: {reward:8.4f} | Portfolio: ${portfolio_after:10.2f} | "
                f"BTC: {self.position:8.4f}"
            )

        return next_state, reward, done, info

    def _get_state(self) -> int:
        state_tuple = self._discretize_features()
        if state_tuple not in self.state_map:
            self.state_map[state_tuple] = self.state_id_counter
            self.state_id_counter += 1
        return self.state_map[state_tuple]

    def _discretize_features(self) -> Tuple:
        state_tuple = []
        for feature in self.feature_cols:
            value = self.df.loc[self.current_step, feature]
            bins = self.feature_bins[feature]
            bin_idx = np.digitize(value, bins) - 1
            bin_idx = np.clip(bin_idx, 0, len(bins) - 2)
            state_tuple.append(bin_idx)
        return tuple(state_tuple)

    def _execute_action(self, action: int, current_price: float) -> None:
        if action == self.ACTION_HOLD:
            return

        if action == self.ACTION_LONG:
            if self.position == 0.0 and self.balance_usd > 0.0:
                gross_usd = self.balance_usd
                fee_paid = gross_usd * self.fee
                net_usd = gross_usd - fee_paid
                btc_bought = net_usd / current_price

                self.position = btc_bought
                self.balance_usd = 0.0
                self.entry_price = current_price

                self.trades.append(
                    {
                        'step': self.current_step,
                        'action': 'BUY',
                        'price': current_price,
                        'amount_btc': btc_bought,
                        'fee_paid_usd': fee_paid,
                    }
                )
            return

        if action == self.ACTION_SHORT:
            if self.position > 0.0:
                btc_to_sell = self.position
                gross_usd = btc_to_sell * current_price
                fee_paid = gross_usd * self.fee
                net_usd = gross_usd - fee_paid

                self.balance_usd = net_usd
                self.position = 0.0
                self.entry_price = 0.0

                self.trades.append(
                    {
                        'step': self.current_step,
                        'action': 'SELL',
                        'price': current_price,
                        'amount_btc': -btc_to_sell,
                        'fee_paid_usd': fee_paid,
                    }
                )

    def _calculate_reward(
        self,
        portfolio_before: float,
        portfolio_after: float,
        action: int,
    ) -> float:
        if portfolio_before > 0:
            return_pct = (portfolio_after - portfolio_before) / portfolio_before
        else:
            return_pct = 0.0

        reward = return_pct
        if action == self.ACTION_HOLD and self.inactivity_penalty > 0:
            reward -= self.inactivity_penalty
        if return_pct < 0:
            reward *= 1.5
        return reward

    def _calculate_portfolio_value(self, current_price: float) -> float:
        btc_value = self.position * current_price
        return self.balance_usd + btc_value

    def get_final_stats(self) -> Dict:
        final_value = self.portfolio_values[-1] if self.portfolio_values else self.initial_balance_usd
        total_return = (final_value - self.initial_balance_usd) / self.initial_balance_usd
        num_trades = len(self.trades)

        if len(self.episode_rewards) > 1:
            sharpe = (
                np.mean(self.episode_rewards)
                / (np.std(self.episode_rewards) + 1e-8)
                * np.sqrt(252 * 24)
            )
        else:
            sharpe = 0.0

        cumulative_max = np.maximum.accumulate(self.portfolio_values)
        drawdown = (np.array(self.portfolio_values) - cumulative_max) / cumulative_max
        max_drawdown = np.min(drawdown) if len(drawdown) > 0 else 0.0

        return {
            'final_portfolio_value': final_value,
            'total_return_pct': total_return * 100,
            'total_trades': num_trades,
            'total_reward': sum(self.episode_rewards),
            'avg_reward': np.mean(self.episode_rewards) if self.episode_rewards else 0.0,
            'max_drawdown': max_drawdown * 100,
            'sharpe_ratio': sharpe,
            'num_steps': len(self.portfolio_values),
        }

    def get_state_space_size(self) -> int:
        return self.state_id_counter

    def get_action_space_size(self) -> int:
        return 3

    def get_observation_space_shape(self) -> int:
        return len(self.feature_cols)


def create_feature_bins(
    train_path: str,
    test_path: str,
    feature_cols: List[str],
    nbins_per_feature: int = 3,
) -> Dict[str, np.ndarray]:
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    combined_df = pd.concat([train_df, test_df], ignore_index=True)

    feature_bins: Dict[str, np.ndarray] = {}
    for feature in feature_cols:
        if feature not in combined_df.columns:
            raise ValueError(f"Feature '{feature}' not found in data.")

        values = combined_df[feature].dropna().values
        quantiles = np.linspace(0, 100, nbins_per_feature + 1)
        bin_edges = np.percentile(values, quantiles)
        bin_edges = np.unique(bin_edges)
        feature_bins[feature] = bin_edges

        print(
            f"Feature {feature:20s} | Bins: {nbins_per_feature} | "
            f"Range: [{bin_edges[0]:.2f}, {bin_edges[-1]:.2f}]"
        )

    return feature_bins


def save_feature_bins(feature_bins: Dict, path: str) -> None:
    bins_serializable = {k: v.tolist() for k, v in feature_bins.items()}
    with open(path, 'w') as f:
        json.dump(bins_serializable, f, indent=2)
    print(f"Saved feature bins: {path}")


def load_feature_bins(path: str) -> Dict[str, np.ndarray]:
    with open(path, 'r') as f:
        bins_dict = json.load(f)
    return {k: np.array(v) for k, v in bins_dict.items()}
