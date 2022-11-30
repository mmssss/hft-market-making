import sys
import os
sys.path.append('../..')

import math
import numpy as np
from collections import OrderedDict
from typing import List, Optional, Tuple, Union, Dict
from simulator_external.simulator.simulator import \
    MdUpdate, Order, OwnTrade, Sim, update_best_positions

"""
References:
     [Stoikov 2008] Avellaneda, M., & Stoikov, S. (2008). High-frequency
     trading in a limit order book. Quantitative Finance, 8(3), 217-224.
"""


class StoikovStrategy:
    """Strategy from [Stoikov 2008]"""

    def __init__(self, sim: Sim, gamma: float, k: float, sigma: float,
                 adjust_delay: int, order_size: float,
                 min_order_size: float, precision: int):
        """
        Args:
            sim:
                Exchange simulator.
            gamma:
                Parameter γ from (29), (30) in [Stoikov 2008]. Assumed to be
                non-negative. Small values correspond to more risk neutral
                strategy, larger values correspond to more risk averse strategy.
            k:
                Parameter k from (30) in [Stoikov 2008]. A statistic that is
                calculated from the market data.
            sigma:
                Parameter σ from (29) in [Stoikov 2008]. A standard deviation
                of increments of the Wiener process that is assumed to be the
                model for the asset price.
            adjust_delay:
                Delay (in nanoseconds) between readjusting the orders.
            order_size:
                Size of limit orders placed by the strategy.
            min_order_size:
                Minimum order size for the base asset allowed by exchange. E.g.
                0.001 BTC for Binance BTC/USDT Perpetual Futures as of 2022-11-29.
            precision:
                Precision of the price - a number of decimal places. E.g. 2 for
                Binance BTC/USDT Perpetual Futures as of 2022-11-29.
        """
        self.sim = sim
        self.gamma = gamma
        self.k = k
        self.sigma = sigma
        self.adjust_delay = adjust_delay
        self.order_size = order_size
        self.min_order_size = min_order_size
        self.precision = precision

        # market data list
        self.md_list: List[MdUpdate] = []
        # executed trades list
        self.trades_list: List[OwnTrade] = []
        # all updates list
        self.updates_list = []
        # all orders list
        self.all_orders = []
        # orders that have not been executed/canceled yet
        self.ongoing_orders: OrderedDict[int, Order] = OrderedDict()
        # current best positions
        self.best_bid = -math.inf
        self.best_ask = math.inf

    def run(self) -> Tuple[List[OwnTrade], List[MdUpdate],
                           List[Union[OwnTrade, MdUpdate]], List[Order]]:
        """This function runs the simulation.

            Args:
            Returns:
                trades_list(List[OwnTrade]):
                    List of our executed trades.
                md_list(List[MdUpdate]):
                    List of market data received by strategy.
                updates_list( List[Union[OwnTrade, MdUpdate]] ):
                    List of all updates received by strategy
                    (market data and information about executed trades).
                all_orders(List[Orted]):
                    List of all placed orders.
        """
        # current position size in base asset
        cur_pos = 0
        # timestamp when last rebalancing happened
        last_readjust = 0

        while True:
            # get update from simulator
            cur_time, updates = self.sim.tick()
            if updates is None:
                break
            # save updates
            self.updates_list += updates
            for update in updates:
                # update best position
                if isinstance(update, MdUpdate):
                    self.best_bid, self.best_ask = update_best_positions(
                        self.best_bid, self.best_ask, update)
                    self.md_list.append(update)
                elif isinstance(update, OwnTrade):
                    own_trade = update
                    if own_trade.side == 'BID':
                        cur_pos += own_trade.size
                    else:
                        cur_pos -= own_trade.size
                    self.trades_list.append(own_trade)
                    # delete executed trades from the dict
                    if own_trade.order_id in self.ongoing_orders.keys():
                        self.ongoing_orders.pop(own_trade.order_id)
                else:
                    assert False, 'invalid type of update!'

            if cur_time - last_readjust > self.adjust_delay:
                last_readjust = cur_time
                # cancel all orders
                while self.ongoing_orders:
                    order_id, _ = self.ongoing_orders.popitem(last=False)
                    self.sim.cancel_order(cur_time, order_id)

                # calculate indifference price, (29) from [Stoikov 2008]
                midprice = (self.best_bid + self.best_ask) / 2
                indiff_price = midprice - cur_pos / self.min_order_size \
                         * self.gamma * self.sigma**2

                # calculate optimal spread, (30) from [Stoikov 2008]
                spread = self.gamma * self.sigma**2 + \
                         2 / self.gamma * math.log(1 + self.gamma / self.k)

                price_bid = round(indiff_price - spread / 2, self.precision)
                price_ask = round(indiff_price + spread / 2, self.precision)
                self.place_order(cur_time, self.order_size, 'BID', price_bid)
                self.place_order(cur_time, self.order_size, 'ASK', price_ask)

        return self.trades_list, self.md_list, self.updates_list, self.all_orders

    def place_order(self, ts: float, size: float, side: str, price: float):
        order = self.sim.place_order(ts, size, side, price)
        self.ongoing_orders[order.order_id] = order
        self.all_orders.append(order)
