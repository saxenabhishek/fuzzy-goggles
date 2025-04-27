from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import (
    EvalCallback,
    StopTrainingOnRewardThreshold,
)
from stable_baselines3.common.env_util import make_vec_env

from src.stock_portfolio_env import StockPortfolioEnv
from src.const import (
    DIR,
    WINDOW_LENGTH,
    TRAINING_ENVS,
    read_dict_from_pickle,
)


CONTINUE_TRAINING = True


if __name__ == "__main__":
    train_df = read_dict_from_pickle("train_df", input_dir=DIR)
    test_df = read_dict_from_pickle("test_df", input_dir="./techLargCapStock")

    test_env = DummyVecEnv(
        [lambda: Monitor(StockPortfolioEnv(test_df, window=WINDOW_LENGTH))]
    )
    test_env = VecNormalize.load("vecnormalize.pkl", test_env)
    test_env.training = False  # Important!
    test_env.norm_reward = False  # Only normalize during training

    train_env = make_vec_env(
        lambda: Monitor(StockPortfolioEnv(train_df, window=WINDOW_LENGTH)),
        n_envs=TRAINING_ENVS,
        seed=42,
    )

    if CONTINUE_TRAINING:
        norm_train_env = VecNormalize.load("./vecnormalize.pkl", train_env)
        model = A2C.load(
            "a2c_stock_trader", norm_train_env, tensorboard_log="./a2c_stock_trader/"
        )
        print("Model Loaded, continuing training")
    else:
        norm_train_env = VecNormalize(
            train_env, norm_obs=True, norm_reward=True, clip_obs=10.0
        )
        model = A2C(
            "MlpPolicy",
            norm_train_env,
            verbose=1,
            tensorboard_log="./a2c_stock_trader/",
        )

    callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=200, verbose=1)

    eval_callback = EvalCallback(
        test_env,
        callback_on_new_best=callback_on_best,
        verbose=1,
        best_model_save_path="./logs/",
        log_path="./logs/",
        eval_freq=500,
        deterministic=True,
        render=False,
    )

    hist = model.learn(
        total_timesteps=50_000,
        progress_bar=True,
        callback=eval_callback,
        tb_log_name="second_run",
        reset_num_timesteps=False,
    )

    model.save("./a2c_stock_trader")
    norm_train_env.save("./vecnormalize.pkl")
    print("model saved, training complete")
