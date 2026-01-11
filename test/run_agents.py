# Run all nopen agents on the test dataset.

import argparse
from pathlib import Path
import json

import numpy as np
import pandas as pd

from config.feature_sets import FEATURE_SETS
from src.environment.trading_env import FeatureTradingEnv, create_feature_bins
from src.agents.q_learning_agent import QLearningAgent
from src.agents.sarsa_agent import SARSAAgent


def parse_agent_id(agent_id: str, algorithm: str) -> dict:
    if not isinstance(agent_id, str):
        return {'feature_set': None, 'fee_label': None, 'penalty_label': None}

    prefix = f"{algorithm}_"
    suffix = agent_id[len(prefix):] if agent_id.startswith(prefix) else agent_id

    parts = suffix.split('_')
    fee_idx = next((i for i, t in enumerate(parts) if t.endswith('pct')), None)

    if fee_idx is None:
        return {'feature_set': suffix, 'fee_label': None, 'penalty_label': None}

    feature_set = '_'.join(parts[:fee_idx])
    fee_label = parts[fee_idx]
    penalty_label = parts[fee_idx + 1] if fee_idx + 1 < len(parts) else None

    return {'feature_set': feature_set, 'fee_label': fee_label, 'penalty_label': penalty_label}


def fee_label_to_float(fee_label: str) -> float:
    if fee_label is None:
        return 0.0
    text = fee_label.replace('pct', '')
    try:
        return float(text) / 100.0
    except ValueError:
        return 0.0


def build_state_map(df: pd.DataFrame, feature_cols: list, feature_bins: dict) -> dict:
    state_map = {}
    state_id_counter = 0

    for idx in range(len(df)):
        state_tuple = []
        for feature in feature_cols:
            value = df.loc[idx, feature]
            bins = feature_bins[feature]
            bin_idx = np.digitize(value, bins) - 1
            bin_idx = np.clip(bin_idx, 0, len(bins) - 2)
            state_tuple.append(int(bin_idx))
        state_tuple = tuple(state_tuple)
        if state_tuple not in state_map:
            state_map[state_tuple] = state_id_counter
            state_id_counter += 1

    return state_map


def evaluate_agent(agent_path: Path, train_path: Path, test_path: Path, feature_bins: dict) -> dict:
    agent_id = agent_path.stem

    if agent_id.startswith('qlearning_'):
        algorithm = 'qlearning'
        agent = QLearningAgent(n_states=100_000, n_actions=3)
    elif agent_id.startswith('sarsa_'):
        algorithm = 'sarsa'
        agent = SARSAAgent(n_states=100_000, n_actions=3)
    else:
        raise ValueError(f"Unknown agent prefix: {agent_id}")

    meta = parse_agent_id(agent_id, algorithm)
    feature_set = meta['feature_set']
    if feature_set not in FEATURE_SETS:
        raise ValueError(f"Unknown feature set: {feature_set}")

    feature_cols = FEATURE_SETS[feature_set]['features']
    fee = fee_label_to_float(meta['fee_label'])

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    state_map = build_state_map(train_df, feature_cols, feature_bins)

    env = FeatureTradingEnv(
        data_path=str(test_path),
        feature_cols=feature_cols,
        feature_bins=feature_bins,
        fee=fee,
        initial_balance=10_000.0,
        inactivity_penalty=0.0,
        verbose=False,
    )

    env.state_map = dict(state_map)
    env.state_id_counter = len(state_map)

    agent.load(str(agent_path))

    state = env.reset()
    done = False
    total_reward = 0.0

    if algorithm == 'sarsa':
        action = agent.choose_action(state, training=False)

    while not done:
        if algorithm == 'qlearning':
            action = agent.choose_action(state, training=False)
            next_state, reward, done, _info = env.step(action)
            state = next_state
        else:
            next_state, reward, done, _info = env.step(action)
            next_action = agent.choose_action(next_state, training=False)
            state = next_state
            action = next_action

        total_reward += reward

    stats = env.get_final_stats()

    return {
        'agent_id': agent_id,
        'algorithm': algorithm,
        'feature_set': feature_set,
        'fee_label': meta['fee_label'],
        'final_portfolio_value': stats['final_portfolio_value'],
        'total_return_pct': stats['total_return_pct'],
        'total_trades': stats['total_trades'],
        'total_reward': total_reward,
        'avg_reward': stats['avg_reward'],
        'max_drawdown': stats['max_drawdown'],
        'sharpe_ratio': stats['sharpe_ratio'],
        'num_steps': stats['num_steps'],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description='Evaluate nopen agents on the test dataset.')
    parser.add_argument('--agents-dir', default='agents', help='Folder with *_nopen.json files')
    parser.add_argument('--data-dir', default='data/splits', help='Folder with btc_train.csv and btc_test.csv')
    parser.add_argument('--output', default='test/agent_test_results.csv', help='Output CSV path')

    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    agents_dir = (repo_root / args.agents_dir).resolve()
    data_dir = (repo_root / args.data_dir).resolve()

    train_path = data_dir / 'btc_train.csv'
    test_path = data_dir / 'btc_test.csv'

    if not agents_dir.exists():
        raise FileNotFoundError(f"Agents dir not found: {agents_dir}")
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(f"Missing train/test CSVs in: {data_dir}")

    agent_files = sorted(agents_dir.glob('*_nopen.json'))
    if not agent_files:
        raise FileNotFoundError(f"No *_nopen.json agents in: {agents_dir}")

    all_feature_cols = list({f for fs in FEATURE_SETS.values() for f in fs['features']})
    feature_bins = create_feature_bins(str(train_path), str(test_path), all_feature_cols, nbins_per_feature=5)

    results = []
    for agent_path in agent_files:
        results.append(evaluate_agent(agent_path, train_path, test_path, feature_bins))

    out_path = (repo_root / args.output).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(results)
    df.to_csv(out_path, index=False)
    print(f"Saved results to: {out_path}")

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
