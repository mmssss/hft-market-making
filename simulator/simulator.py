from dataclasses import dataclass
from typing import Optional
from collections import deque
from queue import PriorityQueue
import os
import math

from tqdm.auto import tqdm
import pandas as pd

# Note: all timestamps are in nanoseconds

@dataclass
class Order:
    """
    Attributes that are note Optional must be not None
    otherwise ValueError is raised.
    """
    # Timestamp of when the order was created in the client time.
    # It can be equal to `receive_ts` of the last update from the strategy
    # returned by method `tick`, or it can be slightly greater,
    # if you want to account for code execution time.
    client_ts: int
    # Timestamp of when the order was created on the exchange
    exchange_ts: Optional[int]
    # ID assigned by client to immideately keep track of the order
    client_order_id: Optional[int]
    # ID assigned by the exchange when the order is placed
    order_id: Optional[int]
    # Side of the trade ('BID' or 'ASK')
    side: str
    size: float
    price: float

    def __post_init__(self):
        if self.client_ts is None:
            raise ValueError('`client_ts` is mandatory')
        if self.side is None:
            raise ValueError('`side` is mandatory')
        if self.size is None:
            raise ValueError('`size` is mandatory')
        if self.price is None:
            raise ValueError('`price` is mandatory')

        if self.side != 'BID' and self.side != 'ASK':
            raise ValueError('Side must be `BID` or `ASK`')
        if self.size <= 0:
            raise ValueError('Order size must be positive')
        if self.price <= 0:
            raise ValueError('Order price must be positive')


@dataclass
class CancelOrder:
    """A query to the simulator to cancel the order by id.

    Attributes that are note Optional must be not None,
    otherwise ValueError is raised.
    """
    # timestamp of when the order was created in the client time.
    client_ts: int
    # timestamp of when the order was created on the exchange
    exchange_ts: Optional[int]
    # exchange assigned ID of the order that needs to be canceled
    order_id: int


@dataclass
class OwnTrade:
    """ Execution of own placed order """
    timestamp: int
    trade_id: int
    order_id: int
    side: str
    size: float
    price: float


@dataclass
class OrderbookSnapshotUpdate:
    """ Orderbook tick snapshot """
    receive_ts:  int  # timestamp of when the data was received by client
    exchange_ts: int  # timestamp on the exchange timeline

    # Each element of the list is a `tuple[price, size]`.
    # First element of the list is the best level, second element
    # is the second best level, and so on.
    asks: list[tuple[float, float]]
    bids: list[tuple[float, float]]


@dataclass
class AnonTrade:
    """ Market trade """
    receive_ts:  int  # timestamp of when the data was received by client
    exchange_ts: int  # timestamp on the exchange timeline
    side: str
    size: float
    price: str


@dataclass
class MdUpdate:
    """ Data of a tick """
    orderbook: Optional[OrderbookSnapshotUpdate] = None
    trade: Optional[AnonTrade] = None


class Strategy:
    def __init__(self, max_position: float) -> None:
        self.max_position = max_position
        # client ID that will be assigned to the next created order
        self.current_id = 1
        self.orderbook = None
        self.current_time = None

    def run(self, sim: 'ExchangeSimulator'):
        while True:
            try:
                # TODO: consider that sim.tick() can also return Order, OwnTrade
                md_update = sim.tick()
                if md_update.orderbook is not None:
                    self.orderbook = md_update.orderbook
                    self.current_time = self.orderbook.receive_ts
                elif md_update.trade is not None:
                    trade = md_update.trade
                    # TODO: update local version of the order book
                    self.current_time = trade.receive_ts
                if self.orderbook is None:
                    continue

                order = Order(self.current_time + 20, None, self.get_current_id(), None,
                              'BID', 0.001, self.orderbook.bids[0][0])
                sim.place_order(order)

            except StopIteration:
                break

    def get_current_id(self):
        cur_id = self.current_id
        self.current_id += 1
        return cur_id


def load_md_from_files(lobs_path: str, trades_path: str) -> list[MdUpdate]:
    """ Loads market data from files.

    This function loads order book snapshots and trades from csv files,
    then creates a list of `MdUpdate` objects, using the data reception timestamps
    for ordering the data in a list.

    Expected order of columns in `lobs.csv`:
    1. timestamp of when the data was received by client (int64)
    2. timestamp on the exchange timeline (int64)
    3. best ask price
    4. best ask volume
    5. best bid price
    6. best bid volume
    7. second best ask price
    8. ...
    and so on.

    Expected order of columns in `trades.csv`:
    1. timestamp of when the data was received by client (int64)
    2. timestamp on the exchange timeline (int64)
    3. side of the trade (string 'BID' or 'ASK')
    4. price
    5. size

    :param lobs_path: path to csv file which contains order book snapshots data
    :param trades_path: path to csv file which contains trades data
    :return: list of `MdUpdate` objects
    """
    print('Loading market data...', end=' ')
    lobs = pd.read_csv(os.path.join(lobs_path))
    trades = pd.read_csv(os.path.join(trades_path))
    print('Done.\n')
    # separate numpy arrays with receive timestamps of type int64
    lobs_ts = lobs[['receive_ts', 'exchange_ts']].to_numpy()
    trades_ts = trades[['receive_ts', 'exchange_ts']].to_numpy()
    # separate numpy arrays with data of type float64
    lobs = lobs.iloc[:, 2:].to_numpy()
    trades = trades.iloc[:, 2:].to_numpy()
    # now create the list of `MdUpdate` objects
    i = 0
    j = 0
    market_data = []
    print('Creating a list of MdUpdate objects...', end=' ')
    with tqdm(total=len(lobs_ts) + len(trades_ts)) as progress_bar:
        while i < len(lobs_ts) and j < len(trades_ts):
            orderbook = None
            trade = None
            # compare receive_ts
            # if equal, both will be created
            if lobs_ts[i][0] <= trades_ts[j][0]:
                # create OrderbookSnapshotUpdate
                asks = []
                bids = []
                for k in range(0, len(lobs[i]), 4):
                    asks.append((lobs[i][k], lobs[i][k+1]))
                    bids.append((lobs[i][k+2], lobs[i][k+3]))
                orderbook = OrderbookSnapshotUpdate(
                    receive_ts=lobs_ts[i][0], exchange_ts=lobs_ts[i][1], asks=asks, bids=bids)
                i += 1
            # compare receive_ts
            elif trades_ts[j][0] <= lobs_ts[i][0]:
                trade = AnonTrade(trades_ts[j][0], trades_ts[j][1], trades[j][0], trades[j][2], trades[j][1])
                j += 1
            market_data.append(MdUpdate(orderbook, trade))
            progress_bar.update(1)
    print('Done.')
    return market_data


class ExchangeSimulator:
    """ Exchange simulator with order book and trades market data.

    This simulator must be used in the following way:

    1. initialize the object of this class, providing parameters and path to market data;
    2. call method `tick` to get market data update, which will contain two timestamps:

       a. `exchange_ts` - when the event was registered on the exchange;
       b. `receive_ts` - when the data was received by client.

    `receive_ts` **should be considered as the current client time.**

    3. call methods `place_order` and `cancel_order` according to your strategy,
       using `receive_ts` from the previous step as the current client time to
       calculate the value of parameter `client_ts` (see `place_order` for details).
    """
    def __init__(self, lobs_path: str, trades_path: str,
                 exec_latency: float, md_latency: float) -> None:
        """
        :param lobs_path: path to csv file which contains order book snapshots data
        :param trades_path: path to csv file which contains trades data
        :param exec_latency: execution latency
        :param md_latency: market data latency
        """
        if exec_latency < 0:
            raise ValueError('Execution latency cannot be negative')
        if md_latency < 0:
            raise ValueError('Market data latency cannot be negative')

        # execution latency
        self.exec_latency = exec_latency
        # market data latency
        self.md_latency = md_latency
        # the ID that will be assigned to the next created order
        self.cur_order_id = 1
        # market data queue
        self.md_queue = load_md_from_files(lobs_path, trades_path)
        # The queue of strategy actions, used to simulate execution latency.
        # Possible types of objects inside the queue: `Order`, `CancelOrder`.
        self.actions_queue = deque()
        # queue for the data that is sent to the strategy
        self.strategy_updates_queue = PriorityQueue()

    def tick(self) -> MdUpdate:
        if not self.md_queue:
            event_time_md = math.inf
        else:
            md_update = self.md_queue[-1]
            if md_update.orderbook is not None:
                event_time_md = md_update.orderbook.exchange_ts
            elif md_update.trade is not None:
                event_time_md = md_update.trade.exchange_ts

        if not self.actions_queue:
            event_time_actions = math.inf
        else:
            event_time_actions = self.actions_queue[-1].client_ts

        if not self.strategy_updates_queue:
            event_time_strat_updates = 


        self._execute_orders()
        self._prepare_orders()

        # return next(self.md)

    def _execute_orders(self):
        pass

    def _prepare_orders(self):
        pass

    def place_order(self, order: 'Order'):
        """Place order.
        :param order: `Order` object. The following attributes MUST be initalized:
            `client_ts`, `side`, `size`, `price`.
        """
        self.actions_queue.append(order)

    def cancel_order(self, order_id):
        """Cancel order by ID.
        :param order_id: order ID assigned by the exchange when the order was placed
        """
        pass


if __name__ == "__main__":
    lobs_path = '../data/1/btcusdt:Binance:LinearPerpetual/lobs.csv'
    trades_path = '../data/1/btcusdt:Binance:LinearPerpetual/trades.csv'

    strategy = Strategy(max_position=0.01)
    sim = ExchangeSimulator(lobs_path, trades_path, exec_latency=10, md_latency=10)
    # strategy.run(sim)

