import numpy as np
import random

# 迷路のサイズを定義
maze_size = 5

# ゴールの位置を設定（例：右下）
goal = (4, 4)

# 迷路の初期状態（0: 通れる場所, 1: 壁）
maze = np.zeros((maze_size, maze_size))
maze[1, 1] = 1  # 壁を設置
maze[1, 3] = 1  # 壁を設置
maze[1, 4] = 1  # 壁を設置
maze[2, 2] = 1  # 壁を設置
maze[3, 1] = 1  # 壁を設置
maze[4, 3] = 1  # 壁を設置
# 必要に応じて壁を追加

# 状態数は迷路の各セル
state_size = maze_size * maze_size

# 行動は上・下・左・右の4つ
action_size = 4

# Qテーブルをゼロで初期化
Q = np.zeros((state_size, action_size))

# 学習パラメータ
alpha = 0.1       # 学習率
gamma = 0.9       # 割引率
epsilon = 0.1     # ε-greedy法のε
episodes = 1000   # エピソード数

def choose_action(state):
    """行動を選択する関数（ε-greedy法）"""
    if random.uniform(0, 1) < epsilon:
        # ランダムに行動を選ぶ（探索）
        return random.randint(0, action_size - 1)
    else:
        # Q値が最大の行動を選ぶ（活用）
        return np.argmax(Q[state])

def state_to_position(state):
    """状態を位置に変換する関数"""
    return (state // maze_size, state % maze_size)

def position_to_state(position):
    """位置を状態に変換する関数"""
    return position[0] * maze_size + position[1]

def get_next_state(state, action):
    """次の状態を取得する関数"""
    x, y = state_to_position(state)

    # 行動に応じて位置を変更
    if action == 0:  # 上
        new_x = x - 1
        new_y = y
    elif action == 1:  # 下
        new_x = x + 1
        new_y = y
    elif action == 2:  # 左
        new_x = x
        new_y = y - 1
    elif action == 3:  # 右
        new_x = x
        new_y = y + 1

    # np.clip を使ってインデックスを範囲内に収める
    new_x = np.clip(new_x, 0, maze_size - 1)
    new_y = np.clip(new_y, 0, maze_size - 1)

    if maze[new_x, new_y] == 1:
        # 壁にぶつかった場合は元の状態に戻る
        return state
    return position_to_state((new_x, new_y))

def get_reward(state):
    """報酬を取得する関数"""
    position = state_to_position(state)
    if position == goal:
        return 100  # ゴールに到達
    else:
        return -1   # 通常の移動はマイナス

def print_maze_with_path(path):
    """迷路と経路を表示する関数"""
    maze_display = maze.copy().astype(str)
    maze_display[maze_display == '0.0'] = '　'  # 通路を空白に
    maze_display[maze_display == '1.0'] = '■'  # 壁を■に
    for pos in path:
        if pos != (0, 0) and pos != goal:
            maze_display[pos] = '・'  # 通過した道を・に
    maze_display[start_position] = 'S'  # スタートをSに
    maze_display[goal] = 'G'   # ゴールをGに
    for row in maze_display:
        print(' '.join(row))

# Q学習のメインループ
for episode in range(episodes):
    # 初期状態をランダムに設定
    current_state = random.randint(0, state_size - 1)

    # ゴールの状態で開始しないようにする
    while current_state == position_to_state(goal):
        current_state = random.randint(0, state_size - 1)

    done = False
    step = 0

    while not done:
        # 行動を選択
        action = choose_action(current_state)

        # 次の状態を取得
        next_state = get_next_state(current_state, action)

        # 報酬を取得
        reward = get_reward(next_state)

        # Q値を更新
        Q[current_state, action] = Q[current_state, action] + alpha * (
            reward + gamma * np.max(Q[next_state]) - Q[current_state, action]
        )

        # 次の状態がゴールなら終了
        if next_state == position_to_state(goal):
            done = True

        # 状態を更新
        current_state = next_state
        step += 1

    # 進捗を表示（オプション）
    if (episode + 1) % 100 == 0:
        print(f"エピソード {episode + 1} 完了")

# 学習結果の確認
start_position = (0, 0)  # スタート位置を設定
start = position_to_state(start_position)
current_state = start
path = [start_position]

while current_state != position_to_state(goal):
    action = np.argmax(Q[current_state])
    next_state = get_next_state(current_state, action)
    if next_state == current_state:
        print("行き止まりに到達しました")
        break
    path.append(state_to_position(next_state))
    current_state = next_state

print("\n最適な経路:")
print(path)

print("\n迷路と最適経路:")
print_maze_with_path(path)
