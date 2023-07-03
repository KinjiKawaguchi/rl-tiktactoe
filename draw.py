import math
import random
import numpy as np
from time import sleep
import matplotlib.pyplot as plt
from IPython.display import clear_output
import time


# プレイヤの入力を受け付ける関数
def get_player_input(play_area, player_turn):
    choosable_area = [str(area) for area in play_area if type(area) is int]

    # 入力待ち
    while(True):
        player_input = input('Choose a number >')
        if player_input in choosable_area:
            player_input = int(player_input)
            break
        else:
            print('Wrong input!\nChoose a number from {}'.format(choosable_area))
    
    # player_turn=1: 先手，player_turn=2: 後手
    if player_turn == 1:
        play_area[play_area.index(player_input)] = '○'
    elif player_turn == 2:
        play_area[play_area.index(player_input)] = '×'
    
    return play_area, player_input

# Q学習AIの行動を決定する関数
def get_ql_action(play_area, choosable_area, q_table, epsilon):
    # 探索行動
    if np.random.rand() < epsilon:
        ai_input = int(random.choice(choosable_area))
    # 貪欲行動
    else:
        row_index = find_q_row(play_area)
        first_choice_flg = 1
        for choice in choosable_area:
            if first_choice_flg == 1:
                ai_input = int(choice)
                first_choice_flg = 0
            else:
                if q_table[row_index, ai_input-1] < q_table[row_index, int(choice)-1]:
                    ai_input = int(choice)
    return ai_input

# AIの入力を受け付ける関数
def get_ai_input(play_area, ai_turn, mode=0, q_table=None, epsilon=None):
    choosable_area = [str(area) for area in play_area if type(area) is int]

    # mode=0: ランダムAI，mode=1: Q学習AI
    if mode == 0:
        ai_input = int(random.choice(choosable_area))
    elif mode == 1:
        ai_input = get_ql_action(play_area, choosable_area, q_table, epsilon)
    
    # ai_turn=1: 先手，ai_turn=2: 後手
    if ai_turn == 1:
        play_area[play_area.index(ai_input)] = '×'
    elif ai_turn == 2:
        play_area[play_area.index(ai_input)] = '○'
    
    return play_area, ai_input

# ゲーム画面を表示する関数
def show_play(play_area):
    clear_output()
    plt.figure(figsize=(6, 6))
    plt.plot()
    plt.xticks([0, 5, 10, 15])
    plt.yticks([0, 5, 10, 15])
    plt.tick_params(labelbottom='off', bottom='off')
    plt.tick_params(labelleft='off', left='off')
    plt.xlim(0, 15)
    plt.ylim(0, 15)

    x_pos = [2.5, 7.5, 12.5]
    y_pos = [2.5, 7.5, 12.5]

    markers = ['$' + str(marker) + '$' for marker in play_area]

    marker_count = 0
    for y in reversed(y_pos):
        for x in x_pos:
            if markers[marker_count] == '$○$':
                color = 'r'
            elif markers[marker_count] == '$×$':
                color = 'k'
            else:
                color = 'b'
            plt.plot(x, y, marker=markers[marker_count], 
                     markersize=30, color=color)
            marker_count += 1
    plt.show()

# ゲーム終了と勝敗を判定する関数
def judge(play_area, inputter):
    end_flg = 0
    winner = 'Nobody'
    first_list = [0, 3, 6, 0, 1, 2, 0, 2]
    second_list = [1, 4, 7, 3, 4, 5, 4, 4]
    third_list = [2, 5, 8, 6, 7, 8, 8, 6]
    for first, second, third in zip(first_list, second_list, third_list):
        if play_area[first] == play_area[second] and play_area[first] == play_area[third]:
            winner = inputter
            end_flg = 1
            break
    choosable_area = [str(area) for area in play_area if type(area) is int]
    if len(choosable_area) == 0:
        end_flg = 1
    return winner, end_flg

# Qテーブルを作成する関数
def make_q_table():
    n_columns = 9
    n_rows = 3**9
    return np.zeros((n_rows, n_columns))

# Qテーブルを更新する関数
def q_learning(play_area, ai_input, reward, play_area_next, q_table, end_flg):
    # 行番号取得
    row_index = find_q_row(play_area)
    row_index_next = find_q_row(play_area_next)
    column_index = ai_input - 1

    # end_flg=1: 勝利or敗北
    if end_flg == 1:
        q_table[row_index, column_index] = q_table[row_index, column_index] + eta * (reward - q_table[row_index, column_index])
    else:
        q_table[row_index, column_index] = q_table[row_index, column_index] + eta * (reward + gamma * np.nanmax(q_table[row_index_next,: ]) - q_table[row_index, column_index])
    
    return q_table

# 状態に対応するQテーブルの行番号を計算する関数
def find_q_row(play_area):
    row_index = 0
    for index in range(len(play_area)):
        if play_area[index] == '○':
            coef = 1
        elif play_area[index] == '×':
            coef = 2
        else:
            coef = 0
        row_index += (3 ** index) * coef
    return row_index

# ランダムAIとQ学習AIのゲームを実行する関数
def randomAI_vs_QLAI(first_inputter, q_table, epsilon=0.5):
    inputter1 = 'Random AI'
    inputter2 = 'Q-Learning AI'

    ql_input_list = []
    play_area_list = []
    play_area = list(range(1, 10))
    inputter_count = first_inputter
    end_flg = 0
    ql_flg = 0
    reward = 0
    game_count = 0 # ゲームの回数をカウントするための変数を追加します
    while True:
        # Q学習退避用
        play_area_tmp = play_area.copy()
        play_area_list.append(play_area_tmp)
        # Q学習実行フラグ
        ql_flg = 0
        # Q学習AIの手番
        if (inputter_count % 2) == 0:
            # Q学習AIの入力
            play_area, ql_ai_input = get_ai_input(play_area, first_inputter, mode=1, q_table=q_table, epsilon=epsilon)
            winner, end_flg = judge(play_area, inputter2)
            # Q学習退避用
            ql_input_list.append(ql_ai_input)            
            # Q学習AIが勝利した場合
            if winner == inputter2:
                reward = 1
                ql_flg = 1
            play_area_before = play_area_list[-1]
            ql_ai_input_before = ql_input_list[-1]
        # ランダムAIの手番
        elif (inputter_count % 2) == 1:
            play_area, random_ai_input = get_ai_input(play_area, first_inputter+1, mode=0)
            winner, end_flg = judge(play_area, inputter1)
            # ランダムAIが勝利した場合
            if winner == inputter1:
                reward = -1
            #ランダムAIが先手の場合の初手以外は学習
            #elif winner == 'Nobody':
            #    reward = -0.5 # 引き分けの場合の報酬を追加します
            if inputter_count != 1:
                ql_flg = 1
        # Q学習実行
        if ql_flg == 1:
            ql_ai_input_before = ql_input_list[-1]
            q_table = q_learning(play_area_before, ql_ai_input_before, reward, play_area, q_table, end_flg)
        if end_flg:
            break
        inputter_count += 1
        game_count += 1 # ゲームの回数をインクリメントします
    #print('{} win!!!'.format(winner))
    return winner, q_table


import multiprocessing as mp
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

start_time = time.time()  # 測定開始

eta = 0.5  # 学習率
gamma = 0.9  # 時間割引率
initial_epsilon = 0.5  # ε
episode = 10000  # エピソード数

# 学習セット数
learning_sets = 1000


def worker(set_index, return_dict):
    winner_list = []
    # Q学習テーブルを初期化
    q_table = make_q_table()

    # 学習プロセス
    for i in range(episode):
        epsilon = initial_epsilon * (episode-i) / episode
        _, q_table = randomAI_vs_QLAI(1, q_table, epsilon)

    # テストプロセス（モデルはこれ以上更新しない）
    test_episodes = 10000  # テストエピソード数を設定
    for i in range(test_episodes):
        winner, _ = randomAI_vs_QLAI(1, q_table, 0)  # ε=0で学習せずに実行
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

    # 勝率の平均と中央値、分散と標準偏差を計算
    mean_win_rate = np.mean(win_rates)
    median_win_rate = np.median(win_rates)
    variance = np.var(win_rates)
    std_deviation = np.std(win_rates)

    print(f"Mean Win Rate: {mean_win_rate}")
    print(f"Median Win Rate: {median_win_rate}")
    print(f"Variance of Win Rate: {variance}")
    print(f"Standard Deviation of Win Rate: {std_deviation}")

    end_time = time.time()  # 測定終了
    elapsed_time = end_time - start_time  # 経過時間（秒）
    print(f"Elapsed time: {elapsed_time} seconds")  # 経過時間を出力

    # 勝率の分布をヒストグラムとして表示
    plt.hist(win_rates, bins=50, alpha=0.5)
    plt.title('Distribution of Win Rates')
    plt.xlabel('Win Rate')
    plt.ylabel('Counts')

    # 平均値と中央値、±1標準偏差を縦線でプロット
    plt.axvline(mean_win_rate, color='r', linestyle='dashed', linewidth=1)
    plt.axvline(median_win_rate, color='g', linestyle='dashed', linewidth=1)
    plt.axvline(mean_win_rate - std_deviation, color='b', linestyle='dashed', linewidth=1)
    plt.axvline(mean_win_rate + std_deviation, color='b', linestyle='dashed', linewidth=1)

    # レジェンドを作成
    plt.legend({'Mean':mean_win_rate, 'Median':median_win_rate, 
                'Mean - STD':mean_win_rate - std_deviation, 'Mean + STD':mean_win_rate + std_deviation})

    plt.show()
