import multiprocessing as mp
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import concurrent.futures


eta = 0.2  # 学習率
gamma = 0.9  # 時間割引率
initial_epsilon = 0.5  # ε
episode = 10000  # エピソード数

# 学習セット数
learning_sets = 100


def worker(set_index):
    # ランダムAI vs Q学習AI
    winner_list = []
    # Q学習テーブルを初期化
    q_table = make_q_table()
    for i in range(episode):
        epsilon = initial_epsilon * (1 - i / episode)
        winner, q_table = randomAI_vs_QLAI(1, q_table, epsilon)
        winner_list.append(winner)

    # 各セットでのQ-Learning AIの勝率を計算し、リストに追加
    ql_win_rate = winner_list.count('Q-Learning AI') / len(winner_list)
    print(f"Learning Set {set_index+1}, Q-Learning AI Win Rate: {ql_win_rate}")
    return ql_win_rate, q_table  # このworkerはQテーブルも返します

def combine_q_tables(q_tables):
    combined_q_table = np.mean(q_tables, axis=0)
    return combined_q_table

if __name__ == "__main__":
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(worker, set_index) for set_index in range(learning_sets)]
        results = [f.result() for f in concurrent.futures.as_completed(futures)]

    # ワーカーから結果（勝率とQテーブル）を分離します
    win_rates, q_tables = zip(*results)

    # Qテーブルを統合します
    combined_q_table = combine_q_tables(q_tables)

    # 新しいQテーブルを使用して、AIとランダムAIを戦わせ、その勝率を計算します
    winner_list = []
    for i in range(10000):
        winner, _ = randomAI_vs_QLAI(1, combined_q_table, epsilon=0.0)  # ε=0で統合されたQテーブルを使用
        winner_list.append(winner)

    ql_win_rate = winner_list.count('Q-Learning AI') / len(winner_list)
    print(f"After combining Q tables, Q-Learning AI Win Rate: {ql_win_rate}")
