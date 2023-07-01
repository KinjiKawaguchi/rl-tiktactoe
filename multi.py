import multiprocessing as mp
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

eta = 0.2  # 学習率
gamma = 0.9  # 時間割引率
initial_epsilon = 0.5  # ε
episode = 10000  # エピソード数

# 学習セット数
learning_sets = 100


def worker(set_index, return_dict):
    winner_list = []
    # Q学習テーブルを初期化
    q_table = make_q_table()
    for i in range(episode):
        epsilon = initial_epsilon * (1 - i / episode)
        winner, _ = randomAI_vs_QLAI(1, q_table, epsilon)
        winner_list.append(winner)
    ql_win_rate = winner_list.count('Q-Learning AI') / len(winner_list)
    print(f"Learning Set {set_index+1}, Q-Learning AI Win Rate: {ql_win_rate}")
    return_dict[set_index] = ql_win_rate


if __name__ == "__main__":
    manager = mp.Manager()
    return_dict = manager.dict()

    processes = []
    for set_index in range(learning_sets):
        p = mp.Process(target=worker, args=(set_index, return_dict))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    win_rates = list(return_dict.values())

    # 勝率の平均と中央値を計算
    mean_win_rate = np.mean(win_rates)
    median_win_rate = np.median(win_rates)

    # 勝率の分布をヒストグラムとして表示
    plt.hist(win_rates, bins=50, range=(0.5, 0.7), density=True, alpha=0.5)

    # 勝率の分布をカーネル密度推定で表示
    sns.kdeplot(win_rates)
    plt.show()