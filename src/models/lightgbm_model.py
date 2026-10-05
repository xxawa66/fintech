"""Fixed CPU LightGBM regression; validation labels do not control training."""
import lightgbm as lgb
import numpy as np
import pandas as pd


def train_model(x: pd.DataFrame, y: np.ndarray, params: dict, rounds: int,
                progress=None) -> lgb.Booster:
    dataset = lgb.Dataset(x, label=y, feature_name=list(x.columns), free_raw_data=True)

    def report(env):
        if progress and ((env.iteration + 1) % 25 == 0 or env.iteration + 1 == rounds):
            progress(f"training rounds {env.iteration + 1}/{rounds}")

    return lgb.train(dict(params), dataset, num_boost_round=rounds, callbacks=[report])
