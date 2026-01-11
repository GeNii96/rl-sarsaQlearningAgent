# Train Q-Learning agents across feature sets and fees.

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.environment.trading_env import FeatureTradingEnv, create_feature_bins, save_feature_bins
from src.agents.q_learning_agent import QLearningAgent
from config.feature_sets import FEATURE_SETS, TRAINING_CONFIGS

NBINS_PER_FEATURE = 5
TRAIN_PATH = str(PROJECT_ROOT / 'data' / 'splits' / 'btc_train.csv')
TEST_PATH = str(PROJECT_ROOT / 'data' / 'splits' / 'btc_test.csv')

OUTPUT_DIR = str(PROJECT_ROOT / 'experiments' / 'qlearning')
AGENTS_DIR = str(PROJECT_ROOT / 'experiments' / 'qlearning' / 'agents')
LOGS_DIR = str(PROJECT_ROOT / 'experiments' / 'qlearning' / 'logs')
FEATURE_BINS_PATH = str(PROJECT_ROOT / 'data' / 'metadata' / f'feature_bins_{NBINS_PER_FEATURE}.json')

Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
Path(AGENTS_DIR).mkdir(parents=True, exist_ok=True)
Path(LOGS_DIR).mkdir(parents=True, exist_ok=True)
Path(FEATURE_BINS_PATH).parent.mkdir(parents=True, exist_ok=True)


def train_single_agent(
    feature_set_name: str,
    feature_cols: list,
    feature_bins: dict,
    fee: float,
    inactivity_penalty_flag: bool,
    n_episodes: int = 500,
    alpha: float = 0.1,
    gamma: float = 0.99,
    epsilon_start: float = 1.0,
    epsilon_decay: float = 0.9995,
    epsilon_min: float = 0.01,
) -> dict:
    fee_str = f"{fee*100:.1f}pct".replace('.0pct', 'pct')
    penalty_str = 'pen' if inactivity_penalty_flag else 'nopen'
    agent_id = f"qlearning_{feature_set_name}_{fee_str}_{penalty_str}"

    env = FeatureTradingEnv(
        data_path=TRAIN_PATH,
        feature_cols=feature_cols,
        feature_bins=feature_bins,
        fee=fee,
        initial_balance=10_000.0,
        inactivity_penalty=0.05 if inactivity_penalty_flag else 0.0,
        verbose=False,
    )

    agent = QLearningAgent(
        n_states=100_000,
        n_actions=3,
        alpha=alpha,
        gamma=gamma,
        epsilon=epsilon_start,
        epsilon_min=epsilon_min,
        epsilon_decay=epsilon_decay,
    )

    episode_rewards = []
    episode_final_values = []
    episode_trades = []
    episode_times = []

    start_time = time.time()

    for episode in tqdm(range(n_episodes), desc=f'Training {agent_id}'):
        episode_start = time.time()
        state = env.reset()
        episode_reward = 0.0
        done = False

        while not done:
            action = agent.choose_action(state, training=True)
            next_state, reward, done, _info = env.step(action)
            agent.update(state, action, reward, next_state, done)
            episode_reward += reward
            state = next_state

        agent.decay_epsilon()
        ep_stats = env.get_final_stats()
        episode_rewards.append(episode_reward)
        episode_final_values.append(ep_stats['final_portfolio_value'])
        episode_trades.append(ep_stats['total_trades'])
        episode_times.append(time.time() - episode_start)

    total_time = time.time() - start_time

    agent_path = f"{AGENTS_DIR}/{agent_id}.json"
    agent.save(agent_path)

    logs = {
        'agent_id': agent_id,
        'feature_set': feature_set_name,
        'nbins_per_feature': NBINS_PER_FEATURE,
        'fee': fee,
        'inactivity_penalty': inactivity_penalty_flag,
        'n_episodes': n_episodes,
        'episode_rewards': episode_rewards,
        'episode_final_values': episode_final_values,
        'episode_trades': episode_trades,
        'total_time_seconds': total_time,
    }
    log_path = f"{LOGS_DIR}/{agent_id}_training.json"
    with open(log_path, 'w') as f:
        json.dump(logs, f, indent=2)

    last_50_rewards = episode_rewards[-50:] if len(episode_rewards) >= 50 else episode_rewards
    last_50_values = episode_final_values[-50:] if len(episode_final_values) >= 50 else episode_final_values
    last_50_trades = episode_trades[-50:] if len(episode_trades) >= 50 else episode_trades

    stats = {
        'agent_id': agent_id,
        'feature_set': feature_set_name,
        'n_features': len(feature_cols),
        'fee': fee,
        'inactivity_penalty': inactivity_penalty_flag,
        'n_episodes': n_episodes,
        'n_states_visited': agent.get_stats()['n_states_visited'],
        'total_reward': sum(episode_rewards),
        'avg_reward_all': float(np.mean(episode_rewards)),
        'avg_reward_last_50': float(np.mean(last_50_rewards)),
        'std_reward_last_50': float(np.std(last_50_rewards)),
        'final_portfolio_value': float(episode_final_values[-1]),
        'avg_value_last_50': float(np.mean(last_50_values)),
        'total_return_pct': (episode_final_values[-1] - 10_000) / 10_000 * 100,
        'avg_trades_all': float(np.mean(episode_trades)),
        'avg_trades_last_50': float(np.mean(last_50_trades)),
        'training_time_seconds': total_time,
        'training_time_minutes': total_time / 60,
    }

    return stats


def main() -> pd.DataFrame:
    all_feature_cols = list({f for fs in FEATURE_SETS.values() for f in fs['features']})

    feature_bins = create_feature_bins(
        TRAIN_PATH,
        TEST_PATH,
        all_feature_cols,
        nbins_per_feature=NBINS_PER_FEATURE,
    )
    save_feature_bins(feature_bins, FEATURE_BINS_PATH)

    all_results = []

    for feature_set_name, feature_config in FEATURE_SETS.items():
        feature_cols = feature_config['features']
        for fee in TRAINING_CONFIGS['fees']:
            for inactivity_penalty in TRAINING_CONFIGS['inactivity_penalty']:
                stats = train_single_agent(
                    feature_set_name=feature_set_name,
                    feature_cols=feature_cols,
                    feature_bins=feature_bins,
                    fee=fee,
                    inactivity_penalty_flag=inactivity_penalty,
                    n_episodes=TRAINING_CONFIGS['n_episodes'],
                    alpha=TRAINING_CONFIGS['alpha'],
                    gamma=TRAINING_CONFIGS['gamma'],
                    epsilon_start=TRAINING_CONFIGS['epsilon_start'],
                    epsilon_decay=TRAINING_CONFIGS['epsilon_decay'],
                    epsilon_min=TRAINING_CONFIGS['epsilon_min'],
                )
                all_results.append(stats)

    results_df = pd.DataFrame(all_results)
    results_path = f"{OUTPUT_DIR}/results_qlearning.csv"
    results_df.to_csv(results_path, index=False)

    return results_df


if __name__ == '__main__':
    main()
