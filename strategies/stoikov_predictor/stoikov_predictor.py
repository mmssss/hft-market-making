import sys
import os

import pandas as pd

sys.path.append('..')
sys.path.append('../..')

from abc import abstractmethod
import math
import numpy as np
from collections import OrderedDict
from typing import List, Optional, Tuple, Union, Dict
from simulator_external.simulator.simulator import \
    MdUpdate, Order, OwnTrade, Sim, update_best_positions
from stoikov.stoikov import StoikovStrategy


class BasePredictor:
    @abstractmethod
    def predict(self, current_time: int):
        """Predict the asset price at some point in the future"""
        pass


class PredictorIdeal(BasePredictor):
    def __init__(self, midprices: pd.DataFrame, time_offset: pd.Timedelta):
        """
        Args:
            midprices:
                Data frame with `mid_price` column, indexed by receive time.
            time_offset:
                Predictions will be made after this amount of time
                from current time.
        """
        self.midprices = midprices
        self.time_offset = time_offset

    def predict(self, current_time: int):
        idx = self.midprices.index.searchsorted(
            pd.Timestamp(current_time) + self.time_offset, side='left'
        )
        if idx != len(self.midprices):
            return self.midprices.iloc[idx].item()
        else:
            return None


class PredictorNoisy(BasePredictor):
    def __init__(self, midprices: pd.DataFrame, time_offset: pd.Timedelta,
                 noise_func):
        """
        Args:
            midprices:
                Data frame with `mid_price` column, indexed by receive time.
            time_offset:
                Predictions will be made after this amount of time
                from current time.
        """
        self.midprices = midprices
        self.time_offset = time_offset
        self.noise_func = noise_func

    def predict(self, current_time: int):
        idx = self.midprices.index.searchsorted(
            pd.Timestamp(current_time) + self.time_offset, side='left'
        )
        if idx != len(self.midprices):
            pred = self.midprices.iloc[idx].item()
            pred *= self.noise_func()
            return pred
        else:
            return None


class StoikovPredictorStrategy(StoikovStrategy):
    def __init__(self, predictor: BasePredictor, **kwargs):
        """

        Args:
            predictor:
            **kwargs:
        """
        self.predictor = predictor
        super().__init__(**kwargs)

    def get_central_price(self):
        return self.predictor.predict(self.cur_time)

