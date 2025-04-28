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

CONTINUE_TRAINING = False
TB_EXP_NAME = f"A2C_{WINDOW_LENGTH}_volume_ma100_portfolio_reward"
TB_LOG_FILE = "D:/Projects/fuzzy-goggles/a2c_stock_trader"
TIMESTEPS = 50_000

print(
    f"Training stating for {TB_EXP_NAME}, { 'continued' if CONTINUE_TRAINING else 'new'}"
)

if __name__ == "__main__":
    train_df = read_dict_from_pickle("train_df", input_dir=DIR)
    test_df = read_dict_from_pickle("test_df", input_dir="./techLargCapStock")

    train_env = make_vec_env(
        lambda: Monitor(StockPortfolioEnv(train_df, window=WINDOW_LENGTH)),
        n_envs=TRAINING_ENVS,
        seed=42,
    )

    test_env = DummyVecEnv(
        [lambda: Monitor(StockPortfolioEnv(test_df, window=WINDOW_LENGTH))]
    )
    if CONTINUE_TRAINING:
        norm_train_env = VecNormalize.load("./vecnormalize.pkl")
        model = A2C.load(
            "a2c_stock_trader.zip", norm_train_env, tensorboard_log=TB_LOG_FILE
        )
        test_env = VecNormalize.load("vecnormalize.pkl")

        print("Model Loaded, continuing training")
    else:
        test_env = VecNormalize(
            test_env, norm_obs=True, norm_reward=True, clip_obs=10.0
        )
        norm_train_env = VecNormalize(
            train_env, norm_obs=True, norm_reward=True, clip_obs=10.0
        )
        model = A2C(
            "MlpPolicy",
            norm_train_env,
            verbose=1,
            tensorboard_log="./a2c_stock_trader/",
        )

    test_env.training = False  # Important!
    test_env.norm_reward = False  # Only normalize during training
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
    input("Press any key to confirm")

    hist = model.learn(
        total_timesteps=TIMESTEPS,
        progress_bar=True,
        callback=eval_callback,
        tb_log_name=TB_EXP_NAME,
        reset_num_timesteps=False,
    )

    model.save("./a2c_stock_trader")
    norm_train_env.save("./vecnormalize.pkl")
    print("model saved, training complete")
