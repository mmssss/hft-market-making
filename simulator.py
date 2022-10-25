import os
import math
import copy
import heapq
from enum import IntEnum
from dataclasses import dataclass
from typing import Optional, Union, Dict, List, Tuple
from collections import deque, OrderedDict

from sortedcontainers import SortedDict
from tqdm.auto import tqdm
import pandas as pd


__all__ = [
    'Order', 'OrderCancel', 'ResponseCode', 'ActionResponse', 'OwnTrade',
    'OrderbookSnapshotUpdate', 'AnonTrade', 'MdUpdate', 'load_md_from_files',
    'ExchangeSimulator'
]


@dataclass
class Order:
    """Information about order.
    Non-optional attributes must be not None, otherwise ValueError is raised.

    Args:
        client_ts:
            Timestamp (in nanoseconds) of when the order was created in the client time.
            It can be equal to `receive_ts` of the last update from the strategy
            returned by method `tick`, or it can be slightly greater,
            if you want to account for code execution time.
        side:
            Side of the trade ('BID' or 'ASK').
        size:
            Order size.
        price:
            Order price.
        exchange_ts:
            Timestamp (in nanoseconds) of when the order was registered on the exchange simulator.
            Filled in by the simulator.
        receive_ts:
            Timestamp (in nanoseconds) of when the data is received by client. Exchange simulator
            fills in this attribute, emulating latency.
        order_id:
            Unique ID assigned by the exchange simulator when the order is placed.
            Filled in by the simulator.
        client_order_id:
            Unique ID assigned by client. Can be used to immediately keep track of the order.
    """
    client_ts: int
    side: str
    size: float
    price: float
    exchange_ts: Optional[int] = None
    receive_ts: Optional[int] = None
    order_id: Optional[int] = None
    client_order_id: Optional[int] = None

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
class OrderCancel:
    """Information about order cancellation.

    Non-optional attributes must not be `None`, otherwise `ValueError` is raised.
    Either `client_order_id` or `order_id` must be provided. If both are provided,
    `order_id` takes precedence.

    Attributes:
        client_ts:
            Timestamp (in nanoseconds) of when the cancel was created in the client time.
        exchange_ts:
            Timestamp (in nanoseconds) of when the cancel was registered on the exchange simulator.
            Filled in by the simulator.
        receive_ts:
            Timestamp (in nanoseconds) of when the data is received by client.
            Simulator fills in this attribute, emulating latency.
        order_id:
            Unique ID that was assigned by the exchange simulator when the order was placed.
        client_order_id:
            Unique ID of the order that was assigned by client.
    """
    client_ts: int
    exchange_ts: Optional[int] = None
    receive_ts: Optional[int] = None
    order_id: Optional[int] = None
    client_order_id: Optional[int] = None

    def __post_init__(self):
        if self.client_ts is None:
            raise ValueError('`client_ts is mandatory')
        if self.client_order_id is None and self.order_id is None:
            raise ValueError('Either `client_order_id` or `order_id` must be provided')


class ResponseCode(IntEnum):
    """Response codes for `ActionResponse` objects
    Attributes:
        OK:
            No errors occurred.
        PLACE_ORDER_INSUFFICIENT_BALANCE:
            Failed to create the order, because account has insufficient balance.
        CANCEL_ORDER_ID_NOT_FOUND:
            Failed to cancel the order, because the order with
            specified `order_id` does not exist.
        CANCEL_ORDER_CLIENT_ID_NOT_FOUND:
            Failed to cancel the order, because the order with
            specified `client_order_id` does not exist.
    """
    OK = 0
    PLACE_ORDER_INSUFFICIENT_BALANCE = 100
    CANCEL_ORDER_ID_NOT_FOUND = 105
    CANCEL_ORDER_CLIENT_ID_NOT_FOUND = 106


@dataclass
class ActionResponse:
    """Simulator response to the client action.

    Attributes:
        action:
            Client action that caused a response.
        code:
            See `ResponseCode`.
    """
    action: Union[Order, OrderCancel]
    code: ResponseCode


@dataclass
class OwnTrade:
    """Market trade of the client of the simulator.

    An object is created by exchange simulator when client's order is filled.

    Attributes:
        exchange_ts:
            Timestamp (in nanoseconds) of when the trade occured on the exchange simulator.
        receive_ts:
            Timestamp (in nanoseconds) of when the data is received by client. Exchange simulator
            fills in this attribute, emulating latency.
        order_id:
            Exchange-assigned ID of the order that has been filled.
        trade_id:
            Exchange-assigned unique ID of the trade.
        side:
            Side of the trade ('BID' or 'ASK').
        size:
            Trade size.
        price:
            Trade price.
        client_order_id:
            Client-assigned unique ID of the order that has been executed.

    """
    exchange_ts: int
    receive_ts:  int
    order_id: int
    trade_id: int
    side: str
    size: float
    price: float
    client_order_id: Optional[int] = None


@dataclass
class OrderbookSnapshotUpdate:
    """Orderbook tick snapshot.

    Attributes:
        exchange_ts:
            Timestamp (in nanoseconds) in the exchange time (as recorded during the real-world
            data collection)
        receive_ts:
            Timestamp (in nanoseconds) of when the data was received by client (as recorded
            during the real-world data collection)
        asks:
            Each element of the list is a tuple `(price, size)`.
            First element of the list is the best level, second element
            is the second-best level, and so on.
        bids:
            Same logic as for asks.
    """
    exchange_ts: int
    receive_ts:  int
    asks: list[tuple[float, float]]
    bids: list[tuple[float, float]]


@dataclass
class AnonTrade:
    """Market trade.

    Attributes:
        exchange_ts:
            Timestamp (in nanoseconds) in the exchange time (as recorded during the real-world
            data collection).
        receive_ts:
            Timestamp (in nanoseconds) of when the data was received by client (as recorded
            during the real-world data collection).
        side:
            Side of the trade ('BID' or 'ASK').
        size:
            Trade size.
        price:
            Trade price.
    """
    exchange_ts: int
    receive_ts:  int
    side: str
    size: float
    price: float


@dataclass
class MdUpdate:
    """Data of a tick"""
    orderbook: Optional[OrderbookSnapshotUpdate] = None
    trade: Optional[AnonTrade] = None


def load_md_from_files(lobs_path: str, trades_path: str,
                       min_ts: Optional[pd.Timestamp] = None,
                       max_ts: Optional[pd.Timestamp] = None) -> deque[MdUpdate]:
    """ Loads market data from files.

    This function loads order book snapshots and trades from csv files,
    then creates a queue of `MdUpdate` objects, ordered by data reception timestamps.

    Expected order (and names) of columns in `lobs.csv`:
    1. timestamp of when the data was received by client (int64) ('receive_ts')
    2. timestamp on the exchange timeline (int64) ('exchange_ts')
    3. best ask price
    4. best ask volume
    5. best bid price
    6. best bid volume
    7. second-best ask price
    8. ...
    and so on.

    Expected order and names of columns in `trades.csv`:
    1. timestamp of when the data was received by client (int64) (`receive_ts`)
    2. timestamp on the exchange timeline (int64) (`exchange_ts`)
    3. side of the trade (string 'BID' or 'ASK')
    4. price of the trade
    5. size of the trade

    Args:
        lobs_path:
            Path to csv file which contains order book snapshots data.
        trades_path:
            Path to csv file which contains trades data.
        min_ts:
            If provided, market data updates with reception timestamps less
            than `min_ts` will not be included in the resulting queue.
        max_ts:
            If provided, market data updates with reception timestamps greater
            than `max_ts` will not be included in the resulting queue.

    Returns:
        A queue of `MdUpdate` objects.
    """
    if min_ts is not None and max_ts is not None and min_ts >= max_ts:
        raise ValueError('`min_ts` must be less than `max_ts`')
    print('Loading market data...')
    lobs = pd.read_csv(os.path.join(lobs_path), skipinitialspace=True)
    trades = pd.read_csv(os.path.join(trades_path), skipinitialspace=True)
    print('Creating a list of MdUpdate objects...', end=' ')

    if min_ts is not None:
        lobs = lobs[lobs['receive_ts'] >= min_ts.value]
        trades = trades[trades['receive_ts'] >= min_ts.value]
    if max_ts is not None:
        lobs = lobs[lobs['receive_ts'] <= max_ts.value]
        trades = trades[trades['receive_ts'] <= max_ts.value]

    # separate numpy arrays with receive timestamps of type int64
    lobs_ts = lobs[['receive_ts', 'exchange_ts']].to_numpy()
    trades_ts = trades[['receive_ts', 'exchange_ts']].to_numpy()
    # separate numpy arrays with data of type float64
    lobs = lobs.iloc[:, 2:].to_numpy()
    trades = trades.iloc[:, 2:].to_numpy()
    # now create the list of `MdUpdate` objects
    i = 0
    j = 0
    market_data = deque()
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
                    exchange_ts=lobs_ts[i][1], receive_ts=lobs_ts[i][0], asks=asks, bids=bids)
                i += 1
            # compare receive_ts
            elif trades_ts[j][0] <= lobs_ts[i][0]:
                trade = AnonTrade(exchange_ts=trades_ts[j][1], receive_ts=trades_ts[j][0],
                                  side=trades[j][0], size=trades[j][2], price=trades[j][1])
                j += 1
            market_data.append(MdUpdate(orderbook, trade))
            progress_bar.update(1)
    print()
    return market_data


class ExchangeSimulator:
    """Exchange simulator on order book level, with latencies emulation

    This simulator must be used in the following way:

    1. initialize the object of this class, providing parameters and path to market data;
    2. call method `tick` to get an update from the simulator. The update will
       contain two timestamps:

       a. `exchange_ts` - when the event was registered on the exchange;
       b. `receive_ts` - when the data was received by client.

    `receive_ts` **should be considered as the current client time.**

    3. call methods `place_order` and `cancel_order` according to your strategy,
       using `receive_ts` from the previous step as the current client time to
       calculate the value of parameter `client_ts` (see `place_order` for details).
    """

    def __init__(self, lobs_path: str, trades_path: str,
                 exec_latency: int, updates_latency: int,
                 account_size: float, fee: float,
                 min_ts: Optional[pd.Timestamp] = None,
                 max_ts: Optional[pd.Timestamp] = None) -> None:
        """
        Args:
            lobs_path:
                Path to csv file which contains order book snapshots data.
            trades_path:
                Path to csv file which contains trades data.
            exec_latency:
                Latency in nanoseconds between the moment when the client executes
                an action and the moment when this action is registered on the exchange.
            updates_latency:
                Latency in nanoseconds between the moment when the action is registered
                in the simulator, and the moment when the client receives an update from
                the simulator.
            account_size:
                Account size in quote asset.
            fee:
                Maker/taker fee (a fraction, not percent).
            min_ts:
                Passed to marked data loader function `load_md_from_files`.
            max_ts:
                Passed to marked data loader function `load_md_from_files`.
        """
        if exec_latency < 0:
            raise ValueError('Execution latency cannot be negative')
        if updates_latency < 0:
            raise ValueError('Market data latency cannot be negative')
        if account_size < 0:
            raise ValueError('Account size cannot be negative')

        self.exec_latency = exec_latency
        self.updates_latency = updates_latency
        self.account_size = account_size
        # TODO: implement separate maker and taker fees
        self.fee = fee

        # Current exchange time (in nanoseconds).
        # Every tick it is set to the smallest event time of three queues:
        # market data, strategy actions, strategy updates
        self.current_time: int = 0
        # Current position size of a client (in base asset)
        self.position_size: float = 0
        # Amount of quote asset that is frozen by open buy orders
        self.frozen_account: float = 0
        # Amount of base asset that is frozen by open sell orders
        self.frozen_position: float = 0
        # Total value (in quote asset) of the account for every tick
        self.value_history: List[float] = []
        # Exchange time, corresponding to `self.value_history` elements
        self.time_history: List[int] = []
        # Current exchange time (in nanoseconds)
        # Market data deqeue. Elements should be popped from the left.
        self.md: deque[MdUpdate] = load_md_from_files(lobs_path, trades_path, min_ts, max_ts)
        # The dequeue of strategy actions, used to simulate execution latency.
        # Elements should be appended to the right, and popped from the left.
        # Possible object types inside the dequeue: `Order`, `OrderCancel`.
        self.actions: deque[Union[Order, OrderCancel]] = deque()
        # Priority queue (using heapq) for the data that is sent to the strategy.
        # Each element in queue is a tuple (update_obj.receive_ts, counter, update_obj),
        # where 'counter' is used to break comparison ties.
        self.strategy_updates: list[Tuple[int, int, Union[MdUpdate, OwnTrade, ActionResponse]]] = []
        # Counter for items added into `self.strategy_updates` queue
        self.strategy_updates_counter: int = 0
        # For now only store best bid and ask levels on the simulator side
        # and assume the client orders are small enough to always be fully executed
        self.best_bid: Optional[float] = None
        self.best_ask: Optional[float] = None
        # The ID that will be assigned to the next client's order
        self.order_id: int = 1
        # The ID that will be assigned to the next client's trade
        self.trade_id: int = 1
        # The dictionary of used client order IDs.
        # Key: client_order_id, Value: `True` if this ID has already been used, `False` otherwise
        self.used_client_ids: Dict[int, bool] = dict()
        # Dictionary for all active client orders.
        # Key: tuple(0, order_id) or tuple(1, client_order_id). Value: `Order` object.
        # Such key structure allow to, for example, cancel the order using either
        # its order_id or client_order_id.
        self.active_orders: Dict[(int, int), Order] = dict()
        # Dict's for active client orders on bid/ask sides, sorted by price.
        # Key: price. Value: another dict, used to store orders.
        # OrderedDict is used to pop elements with `popitem` in FIFO order.
        #     This ordered dict contains elements as follows.
        #     Key: tuple(0, order_id) or tuple(1, client_order_id). Value: `Order` object.
        self.active_orders_ask: SortedDict[int, OrderedDict[(int, int), Order]] = SortedDict()
        self.active_orders_bid: SortedDict[int, OrderedDict[(int, int), Order]] = SortedDict()
        # Whether method `tick` was called during this simulation
        self.tick_called: bool = False
        self.progress_bar = tqdm(total=len(self.md))

    def tick(self) -> Union[MdUpdate, OwnTrade, ActionResponse, None]:
        """Gets update from the simulator.

        Returns:
            Update from the simulator. Possible return types:
            `MdUpdate`:
                Market data update.
            `OwnTrade`:
                Information about executed trade of the client.
            `ActionResponse`:
                Response to client's action (placing or canceling an order).
                Contains the copy of the action object that was passed to the simulator,
                with extra filled attributes. Also contains execution status and possibly
                the error code. See `Order`, `OrderCancel`, `Response`.
            `None`:
                If the simulation is finished.
        """
        if not self.tick_called:
            self.tick_called = True

        while True:
            md_et = self._get_md_event_time()
            actions_et = self._get_actions_event_time()
            updates_et = self._get_strat_updates_event_time()
            if md_et == math.inf and actions_et == math.inf and updates_et == math.inf:
                return None

            # Return an update to the strategy
            if updates_et < md_et and updates_et < actions_et:
                self.current_time = updates_et
                update = heapq.heappop(self.strategy_updates)[-1]
                return update

            # Apply an update from market data queue
            elif md_et <= updates_et and md_et <= actions_et:
                self.current_time = md_et
                md_update = self.md.popleft()
                self.progress_bar.update(1)
                if md_update.orderbook is not None:
                    self.best_bid = md_update.orderbook.bids[0][0]
                    self.best_ask = md_update.orderbook.asks[0][0]
                    receive_ts = md_update.orderbook.receive_ts
                elif md_update.trade is not None:
                    if md_update.trade.side == 'BID':
                        self.best_ask = md_update.trade.price
                    else:
                        self.best_bid = md_update.trade.price
                    receive_ts = md_update.trade.receive_ts
                self._push_strategy_update(receive_ts, md_update)
                self._execute_orders()

            # Apply an update from actions queue.
            elif actions_et <= updates_et and actions_et <= md_et:
                self.current_time = actions_et
                action = self.actions.popleft()
                # Place order action has waited the execution latency
                if type(action) == Order:
                    order = action
                    # Check if current account size is sufficient to create the order
                    quote_size = order.size * order.price
                    if order.side == 'BID':
                        if quote_size <= self.account_size:
                            self.account_size -= quote_size
                            self.frozen_account += quote_size
                        else:
                            self._push_action_response(order, ResponseCode.PLACE_ORDER_INSUFFICIENT_BALANCE)
                            continue
                    else:
                        # TODO: (1) maybe freeze a part of the position when selling and check if balance
                        #  is sufficient, while taking short-selling into consideration
                        pass
                    # If balance is enough, finalize the order
                    order.order_id = self._next_order_id()
                    if order.side == 'BID':
                        orders = self.active_orders_bid.setdefault(order.price, OrderedDict())
                    else:
                        orders = self.active_orders_ask.setdefault(order.price, OrderedDict())
                    # use `order_id` as key
                    orders[(0, order.order_id)] = order
                    self.active_orders[(0, order.order_id)] = order
                    # use `client_order_id` as key
                    if order.client_order_id is not None:
                        if order.client_order_id not in self.used_client_ids:
                            self.used_client_ids[order.client_order_id] = True
                            orders[(1, order.client_order_id)] = order
                            self.active_orders[(1, order.client_order_id)] = order
                        else:
                            raise RuntimeError(f'Client order id {order.client_order_id} has already been used')
                    self._push_action_response(order, ResponseCode.OK)

                # Cancel order action has waited the execution latency
                elif type(action) == OrderCancel:
                    cancel = action
                    if cancel.order_id is not None:
                        key1 = (0, cancel.order_id)
                        if key1 in self.active_orders:
                            key2 = (1, self.active_orders[key1].client_order_id)
                        else:
                            self._push_action_response(cancel, ResponseCode.CANCEL_ORDER_ID_NOT_FOUND)
                            continue
                    elif cancel.client_order_id is not None:
                        key1 = (1, cancel.client_order_id)
                        if key1 in self.active_orders:
                            key2 = (0, self.active_orders[key1].order_id)
                        else:
                            self._push_action_response(cancel, ResponseCode.CANCEL_ORDER_CLIENT_ID_NOT_FOUND)
                            continue
                    else:
                        raise RuntimeError('Either `client_order_id` or `order_id` must '
                                           'be provided in `OrderCancel` object')

                    order = self.active_orders.pop(key1)
                    if order.side == 'BID':
                        quote_size = order.size * order.price
                        self.frozen_account -= quote_size
                        self.account_size += quote_size
                        active_orders_bidask = self.active_orders_bid
                    else:
                        # TODO: (2) maybe freeze a part of the position when selling and check if balance
                        #  is sufficient, while taking short-selling into consideration
                        active_orders_bidask = self.active_orders_ask
                    active_orders_bidask[order.price].pop(key1)
                    if key2 is not None and key2 != key1:
                        active_orders_bidask[order.price].pop(key2)
                        self.active_orders.pop(key2)
                    self._push_action_response(cancel, ResponseCode.OK)
            mid_price = (self.best_ask - self.best_bid) / 2
            cur_value = self.account_size + self.position_size * mid_price
            self.value_history.append(cur_value)
            self.time_history.append(self.current_time)

    def _get_md_event_time(self):
        # Calculates event time for market data queue.
        # It is equal to `exchange_ts` of the first element in the queue.
        if not self.md:
            event_time = math.inf
        else:
            md_update = self.md[0]
            if md_update.orderbook is not None:
                event_time = md_update.orderbook.exchange_ts
            elif md_update.trade is not None:
                event_time = md_update.trade.exchange_ts
        return event_time

    def _get_actions_event_time(self):
        # Calculates event time for actions queue.
        # It is equal to `exchange_ts` of the first element in the cqueue.
        if not self.actions:
            event_time = math.inf
        else:
            event_time = self.actions[0].exchange_ts
        return event_time

    def _get_strat_updates_event_time(self):
        # Calculates event time for strategy updates queue.
        # It is equal to `receive_ts` of the first element in the queue.
        if not self.strategy_updates:
            event_time = math.inf
        else:
            update = self.strategy_updates[0][-1]
            if type(update) == MdUpdate:
                if update.orderbook is not None:
                    event_time = update.orderbook.receive_ts
                elif update.trade is not None:
                    event_time = update.trade.receive_ts
            elif type(update) == ActionResponse:
                event_time = update.action.receive_ts
            else:  # type(update) is one of [Order, OrderCancel]
                event_time = update.receive_ts
        return event_time

    def _next_order_id(self):
        # Returns the next order ID
        cur_id = self.order_id
        self.order_id += 1
        return cur_id

    def _push_strategy_update(self, receive_ts, update: Union[MdUpdate, OwnTrade, ActionResponse]):
        self.strategy_updates_counter += 1
        heapq.heappush(self.strategy_updates, (receive_ts, self.strategy_updates_counter, update))

    def _push_action_response(self, action: Union[Order, OrderCancel], code: ResponseCode):
        action_copy = copy.deepcopy(action)
        action_copy.receive_ts = action_copy.exchange_ts + self.updates_latency
        response = ActionResponse(action_copy, code)
        self._push_strategy_update(action_copy.receive_ts, response)
        self._execute_orders()

    def _execute_orders(self):
        for price in self.active_orders_ask:
            if price < self.best_bid:
                while self.active_orders_ask[price]:
                    self._execute_orders_on_price_level(self.active_orders_ask, price)
            else:
                break
        for price in reversed(self.active_orders_bid):
            if price > self.best_ask:
                while self.active_orders_bid[price]:
                    self._execute_orders_on_price_level(self.active_orders_bid, price)
            else:
                break

    def _execute_orders_on_price_level(self, active_orders_bidask, price):
        # We know that the key is `order_id`, because we first add the element in the
        # dictionary using `order_id` as the key.
        _, order = active_orders_bidask[price].popitem(last=False)
        if order.side == 'BID':
            self.frozen_account -= order.size * order.price
            self.position_size += (1 - self.fee) * order.size
        else:
            # TODO: (3) maybe freeze a part of the position when selling and check if balance
            #  is sufficient, while taking short-selling into consideration
            self.position_size -= order.size
            self.account_size += order.size * order.price

        self.active_orders.pop((0, order.order_id))
        if order.client_order_id is not None:
            active_orders_bidask[order.price].pop((1, order.client_order_id))
            self.active_orders.pop((1, order.client_order_id))

        receive_ts = self.current_time + self.updates_latency
        trade_id = self._next_trade_id()
        trade = OwnTrade(self.current_time, receive_ts, order.order_id,
                         trade_id, order.side, order.size, order.price,
                         order.client_order_id)
        self._push_strategy_update(receive_ts, trade)

    def _next_trade_id(self):
        # Returns the next trade ID
        cur_id = self.trade_id
        self.trade_id += 1
        return cur_id

    def place_order(self, order: Order):
        """Places the order in exchange simulator, with latency emulation.

        A copy of passed `Order` object is created.

        Args:
            order:
                `Order` object. See `Order` for information about the attributes.
                The same `client_order_id` can only be used once per simulation.

        Raises:
            RuntimeError:
                If method `tick` has never been called before.
            RuntimeError:
                If tried to place an order with client order ID that has already
                been used. This situation does not depend on emulated latency and is caused
                by a client failing to generate unique IDs for the orders. Thus, it makes
                more sense to raise Python error instead of returning appropriate `ActionResponse`.
        """
        if not self.tick_called:
            raise RuntimeError('You need to call `tick` before calling '
                               '`place_order` for the first time')
        if order.client_order_id in self.used_client_ids:
            raise RuntimeError(f'Client order id {order.client_order_id} has already been used')
        order_copy = copy.deepcopy(order)
        order_copy.exchange_ts = order_copy.client_ts + self.exec_latency
        self.actions.append(order_copy)

    def cancel_order(self, order_cancel: OrderCancel):
        """Cancels the order in exchange simulator, with latency emulation.

        A copy of passed `OrderCancel` object is created.

        Args:
            order_cancel:
                `OrderCancel` object. See `OrderCancel` for information about the attributes.

        Raises:
            RuntimeError:
                If method `tick` has never been called before.
        """
        if not self.tick_called:
            raise RuntimeError('You need to call `tick` before calling '
                               '`cancel_order` for the first time')
        cancel_copy = copy.deepcopy(order_cancel)
        cancel_copy.exchange_ts = cancel_copy.client_ts + self.exec_latency
        self.actions.append(cancel_copy)

    def get_value_history(self) -> pd.DataFrame:
        """Returns the history of account value.

        Account value is calculated as follows:
        account_size + position_size * mid_price

        Returns:
            History of account value for every tick.
        """
        df = pd.DataFrame()
        df['exchange_ts'] = self.time_history
        df['exchange_ts'] = pd.to_datetime(df['exchange_ts'])
        df['account_value'] = self.value_history
        return df
