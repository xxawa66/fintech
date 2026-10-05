"""Fixed CPU LightGBM regression; validation labels do not control training."""
from pathlib import Path

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


def save_model(model: lgb.Booster, path: Path) -> None:
    """Use Python file I/O so Windows paths containing Chinese work reliably."""
    path.write_text(model.model_to_string(), encoding="utf-8")


def load_model(path: Path) -> lgb.Booster:
    """Load the same native model format without native narrow-path file I/O."""
    return lgb.Booster(model_str=path.read_text(encoding="utf-8"))
