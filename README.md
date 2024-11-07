# このコード

Q学習がよくわからなかったので動きを確認するために生成した。

QことQテーブルはパネルごとに評価ポイントがわりふられるイメージ

学習ループでは
```
        # Q値を更新
        Q[current_state, action] = Q[current_state, action] + alpha * (
            reward + gamma * np.max(Q[next_state]) - Q[current_state, action]
        )
```

それまでの行動にたいしても値をいれつつ、ゴールするとそれまでの行動にたいしても報酬をわたす。


