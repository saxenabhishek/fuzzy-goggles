from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3 import A2C
from stock_portfolio_env import StockPortfolioEnv
import matplotlib.pyplot as plt
import numpy as np
from const import read_dict_from_pickle, DIR, WINDOW_LENGTH


def plot_and_save_results(
    test_df,
    buy_signals,
    sell_signals,
    rewards,
    profit
):
    plt.figure(figsize=(16, 10))  # Set figure size

    for key in test_df.keys():
        plt.plot(
            test_df[key]["date"],
            test_df[key]["close"],
            label=f"{key} Close Price",
            linewidth=1.2,
        )
        data = test_df[key]
        buy_prices = [
            data[data["date"] == d]["close"].values[0] for d in buy_signals[key]
        ]
        sell_prices = [
            data[data["date"] == d]["close"].values[0] for d in sell_signals[key]
        ]
        plt.scatter(buy_signals[key], buy_prices, color="green", marker="^")
        plt.scatter(sell_signals[key], sell_prices, color="red", marker="v")

    plt.plot(
        test_df[key]["date"],
        np.concatenate([np.zeros(100), profit]),
        label="Profit",
        linewidth=2,
        color="blue",
    )
    plt.plot(
        test_df[key]["date"],
        np.concatenate([np.zeros(100), np.multiply(rewards, 10)]),
        label="Rewards (scaled)",
        linewidth=2,
        color="orange",
    )

    plt.title(
        f"Trading Performance\nTotal Reward: {total_reward:.2f} | Final Profit: {profit[-1]:.2f} | Profit Rate: {info[0]['profit_rate']:.2%}",
        fontsize=16,
    )
    plt.xlabel("Date", fontsize=14)
    plt.ylabel("Price / Profit / Rewards", fontsize=14)
    plt.xticks(fontsize=12, rotation=25)
    plt.yticks(fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(alpha=0.5)
    plt.tight_layout()  # Adjust layout to prevent overlap
    plt.savefig("result.png", dpi=300)  # Save with high resolution
    plt.show()  # Display the plot


if __name__ == "__main__":
    test_df = read_dict_from_pickle("test_df", input_dir=DIR)

    for sym in test_df.keys():
        test_df[sym] = test_df[sym].sort_values(by="date").reset_index(drop=True)

    test_env = DummyVecEnv(
        [lambda: Monitor(StockPortfolioEnv(test_df, window=WINDOW_LENGTH))]
    )
    test_env = VecNormalize.load("vecnormalize.pkl", test_env)
    test_env.training = False  # Important!
    test_env.norm_reward = False  # Only normalize during training

    # Load best model
    eval_model = A2C.load("logs/best_model", test_env)

    obs = test_env.reset()
    done = False
    total_reward = 0
    rewards = []
    profit = []
    while not done:
        action, _state = eval_model.predict(obs, deterministic=True)
        profit.append(test_env.get_attr("total_profit")[0])

        new_obs, reward, done, info = test_env.step(action)
        rewards.append(reward[0])
        total_reward += reward[0]

        obs = new_obs

    test_env.close()
    print(f"Total Reward: {total_reward: .4f}")
    print(f"Final Profit: {profit[-1]: .4f}")
    print(f"Profit Rate {info[0]['profit_rate']: .4f}")

    buy_signals = info[0]['buy_signals']
    sell_signals = info[0]['sell_signals']

    plot_and_save_results(test_df,  buy_signals, sell_signals, rewards, profit)
