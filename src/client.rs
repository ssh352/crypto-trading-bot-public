use crate::{
    algo::{PairsTradingAlgorithm, Signal},
    constants::*,
    event::{Event, OrderBookEventType, OrderBookUpdate, Trade},
    order::{OrderStatus, OrderUpdate},
    order_book_snapshots::OrderBookSnapshots,
    portfolio::Portfolio,
    Exchange::{self, *},
};
use chrono::{DateTime, Duration, Utc};
use crossbeam::channel::{Receiver, Sender};
use std::collections::VecDeque;

pub struct Client {
    to_exchange: Sender<Event>,
    from_exchange: Receiver<Event>,

    // Internal State
    i: i32,
    pf: Portfolio,
    algo: PairsTradingAlgorithm,
    first_bitmex_order_book_event_ts: Option<DateTime<Utc>>,
    first_deribit_order_book_event_ts: Option<DateTime<Utc>>,
    first_bitmex_trade_event_ts: Option<DateTime<Utc>>,
    first_deribit_trade_event_ts: Option<DateTime<Utc>>,
    can_start_ordering: bool,
    bitmex_obs: OrderBookSnapshots,
    deribit_obs: OrderBookSnapshots,
    bitmex_trades: VecDeque<Trade>,
    deribit_trades: VecDeque<Trade>,
    latest_bitmex_market_event: Option<Event>,
    latest_deribit_market_event: Option<Event>,
}

impl Client {
    pub fn new(to_exchange: Sender<Event>, from_exchange: Receiver<Event>) -> Self {
        Self {
            to_exchange,
            from_exchange,

            // Internal state
            i: 0,
            pf: Portfolio::new(),
            algo: PairsTradingAlgorithm::new(),
            first_bitmex_order_book_event_ts: None,
            first_deribit_order_book_event_ts: None,
            first_bitmex_trade_event_ts: None,
            first_deribit_trade_event_ts: None,
            can_start_ordering: false,
            bitmex_obs: OrderBookSnapshots::new(),
            deribit_obs: OrderBookSnapshots::new(),
            bitmex_trades: VecDeque::new(),
            deribit_trades: VecDeque::new(),
            latest_bitmex_market_event: None,
            latest_deribit_market_event: None,
        }
    }

    pub fn run(mut self) {
        loop {
            self.i += 1;

            let event = self.from_exchange.recv().unwrap();

            if let (Some(ts), Some(bm_top), Some(db_top)) = (
                event.timestamp(),
                self.bitmex_obs.ob.top(),
                self.deribit_obs.ob.top(),
            ) {
                print!(
                    "\r{} {} - Bitmex: ({:.2}, {:.2}) Deribit: ({:.2}, {:.2}) Portfolio: ({}, {}) BTC: {:.2} USD/Year: {:.2}",
                    self.i,
                    ts.format("%Y-%m-%d %H:%M:%S"),
                    bm_top.0 .0.as_f64(),
                    bm_top.1 .0.as_f64(),
                    db_top.0 .0.as_f64(),
                    db_top.1 .0.as_f64(),
                    self.pf.current_bitmex_filled_pos(),
                    self.pf.current_deribit_filled_pos(),
                    self.pf.profit_in_btc(),
                    self.pf.profit_per_year_in_usd(*ts)
                );
            }

            match event {
                Event::Market {
                    exchange,
                    bid_price: _,
                    ask_price: _,
                    timestamp: _,
                } if exchange == Deribit => {
                    self.handle_market_event(event);
                }

                Event::OrderBook {
                    exchange,
                    type_,
                    bids,
                    asks,
                    timestamp,
                } => self.handle_order_book_event(exchange, type_, bids, asks, timestamp),

                Event::Trade {
                    exchange,
                    trades,
                    timestamp,
                } => self.handle_trade_event(exchange, trades, timestamp),

                Event::OrderUpdate(ou) => self.handle_order_update(ou),

                err => panic!("{:?}", err),
            }
        }
    }

    fn handle_market_event(&mut self, market_event: Event) {
        // Sanity check that param is really a market event, and update latest market event variables
        match market_event {
            Event::Market {
                exchange,
                bid_price: _,
                ask_price: _,
                timestamp: _,
            } => match exchange {
                Bitmex => self.latest_bitmex_market_event = Some(market_event),
                Deribit => self.latest_deribit_market_event = Some(market_event),
            },
            err => panic!("{:?}", err),
        }

        if !self.can_start_ordering {
            self.to_exchange.send(Event::Signal(Signal::None)).unwrap();
            return;
        }

        if let (Some(latest_bitmex_market_event), Some(latest_deribit_market_event)) = (
            &self.latest_bitmex_market_event,
            &self.latest_deribit_market_event,
        ) {
            let (
                // Latest bid and ask prices from latest market events
                latest_bitmex_bid_price,
                latest_bitmex_ask_price,
                latest_bitmex_market_event_ts,
                latest_deribit_bid_price,
                latest_deribit_ask_price,
                latest_deribit_market_event_ts,
            ) = match (latest_bitmex_market_event, latest_deribit_market_event) {
                (
                    Event::Market {
                        bid_price: latest_bitmex_bid_price,
                        ask_price: latest_bitmex_ask_price,
                        timestamp: latest_bitmex_market_event_ts,
                        ..
                    },
                    Event::Market {
                        bid_price: latest_deribit_bid_price,
                        ask_price: latest_deribit_ask_price,
                        timestamp: latest_deribit_market_event_ts,
                        ..
                    },
                ) => (
                    latest_bitmex_bid_price,
                    latest_bitmex_ask_price,
                    latest_bitmex_market_event_ts,
                    latest_deribit_bid_price,
                    latest_deribit_ask_price,
                    latest_deribit_market_event_ts,
                ),
                err => panic!("{:?}", err),
            };

            // TODO: Currently fills the order immediately at latest market event priceon the client side, when instead should be handled on the exchange and using tick events
            let close_unfilled_ou = if self.pf.is_one_filled_one_unfilled() {
                let latest_bo = self.pf.bitmex_orders.last().unwrap();
                let latest_bou = latest_bo.updates.last().unwrap();
                let latest_do = self.pf.deribit_orders.last().unwrap();
                let latest_dou = latest_do.updates.last().unwrap();

                let bitmex_is_long = self.pf.bitmex_orders.last().unwrap().quantity > 0;

                if bitmex_is_long {
                    if latest_bou.status == OrderStatus::Filled {
                        // Deribit unfilled
                        let long_delta = PairsTradingAlgorithm::long_delta(
                            latest_bo.updates.first().unwrap().ask_price,
                            *latest_deribit_bid_price,
                        );
                        // println!("1 = {:?}", 1);
                        // println!("long_delta = {:?}", long_delta);
                        if long_delta > -CLOSE_UNFILLED_DELTA_THRESHOLD {
                            Some(OrderUpdate {
                                bid_price: *latest_deribit_bid_price,
                                ask_price: *latest_deribit_ask_price,
                                price: *latest_deribit_bid_price + ORDER_PRICE_OFFSET,
                                status: OrderStatus::Filled,
                                timestamp: *latest_deribit_market_event_ts,
                                ..latest_dou.clone()
                            })
                        } else {
                            None
                        }
                    } else {
                        // Bitmex unfilled
                        let long_delta = PairsTradingAlgorithm::long_delta(
                            *latest_bitmex_ask_price,
                            latest_do.updates.first().unwrap().bid_price,
                        );
                        // println!("2 = {:?}", 2);
                        // println!("long_delta = {:?}", long_delta);
                        if long_delta > -CLOSE_UNFILLED_DELTA_THRESHOLD {
                            Some(OrderUpdate {
                                bid_price: *latest_bitmex_bid_price,
                                ask_price: *latest_bitmex_ask_price,
                                price: *latest_bitmex_ask_price - ORDER_PRICE_OFFSET,
                                status: OrderStatus::Filled,
                                timestamp: *latest_bitmex_market_event_ts,
                                ..latest_bou.clone()
                            })
                        } else {
                            None
                        }
                    }
                } else {
                    if latest_bou.status == OrderStatus::Filled {
                        // Deribit unfilled
                        let short_delta = PairsTradingAlgorithm::short_delta(
                            latest_bo.updates.first().unwrap().bid_price,
                            *latest_deribit_ask_price,
                        );
                        // println!("3 = {:?}", 3);
                        // println!("short_delta = {:?}", short_delta);
                        if short_delta < CLOSE_UNFILLED_DELTA_THRESHOLD {
                            Some(OrderUpdate {
                                bid_price: *latest_deribit_bid_price,
                                ask_price: *latest_deribit_ask_price,
                                price: *latest_deribit_ask_price - ORDER_PRICE_OFFSET,
                                status: OrderStatus::Filled,
                                timestamp: *latest_deribit_market_event_ts,
                                ..latest_dou.clone()
                            })
                        } else {
                            None
                        }
                    } else {
                        // Bitmex unfilled
                        let short_delta = PairsTradingAlgorithm::short_delta(
                            *latest_bitmex_bid_price,
                            latest_do.updates.first().unwrap().ask_price,
                        );
                        // println!("4 = {:?}", 4);
                        // println!("short_delta = {:?}", short_delta);
                        if short_delta < CLOSE_UNFILLED_DELTA_THRESHOLD {
                            Some(OrderUpdate {
                                bid_price: *latest_bitmex_bid_price,
                                ask_price: *latest_bitmex_ask_price,
                                price: *latest_bitmex_bid_price + ORDER_PRICE_OFFSET,
                                status: OrderStatus::Filled,
                                timestamp: *latest_bitmex_market_event_ts,
                                ..latest_bou.clone()
                            })
                        } else {
                            None
                        }
                    }
                }
            } else {
                None
            };

            if let Some(ou) = close_unfilled_ou {
                let ou_clone = ou.clone();
                println!(
                        "\r[{}] {:3} * {:7} @ {:7} Failed                                                                                ",
                        ou_clone.timestamp.format("%Y-%m-%d %H:%M:%S"),
                        ou_clone.quantity,
                        ou_clone.exchange.to_string().as_str(),
                        ou_clone.price.to_string(),
                    );
                self.pf.on_order_update(ou.clone());
                self.to_exchange
                    .send(Event::Signal(Signal::PriceAdjustment(ou)))
                    .unwrap();
            } else {
                let signal = self.algo.generate_signal(
                    &self.pf,
                    latest_bitmex_market_event,
                    latest_deribit_market_event,
                );
                if let Signal::NewTrade(bou, dou) = signal.clone() {
                    self.pf.on_order_update(bou);
                    self.pf.on_order_update(dou);
                }

                self.to_exchange.send(Event::Signal(signal)).unwrap();
            }
        }
    }

    fn handle_order_book_event(
        &mut self,
        exchange: Exchange,
        type_: OrderBookEventType,
        bids: Vec<OrderBookUpdate>,
        asks: Vec<OrderBookUpdate>,
        timestamp: DateTime<Utc>,
    ) {
        match exchange {
            Bitmex => {
                if self.first_bitmex_order_book_event_ts == None {
                    self.first_bitmex_order_book_event_ts = Some(timestamp);
                }
                if !self.can_start_ordering {
                    self.check_if_can_start_ordering(timestamp);
                }

                self.bitmex_obs.update(type_, bids, asks, timestamp);

                // For Bitmex, derive a market event from this order book event
                let (bid_price, ask_price) = match self.bitmex_obs.ob.top() {
                    Some(((bid_price, _), (ask_price, _))) => (*bid_price, *ask_price),
                    None => unreachable!(),
                };

                self.handle_market_event(Event::Market {
                    exchange,
                    bid_price,
                    ask_price,
                    timestamp,
                });
            }
            Deribit => {
                if self.first_deribit_order_book_event_ts == None {
                    self.first_deribit_order_book_event_ts = Some(timestamp);
                }
                if !self.can_start_ordering {
                    self.check_if_can_start_ordering(timestamp);
                }

                self.deribit_obs.update(type_, bids, asks, timestamp);
            }
        }
    }

    fn handle_trade_event(
        &mut self,
        exchange: Exchange,
        trades: Vec<Trade>,
        timestamp: DateTime<Utc>,
    ) {
        match exchange {
            Bitmex => {
                if self.first_bitmex_trade_event_ts == None {
                    self.first_bitmex_trade_event_ts = Some(timestamp);
                }
                if !self.can_start_ordering {
                    self.check_if_can_start_ordering(timestamp);
                }
            }
            Deribit => {
                if self.first_deribit_trade_event_ts == None {
                    self.first_deribit_trade_event_ts = Some(timestamp);
                }
                if !self.can_start_ordering {
                    self.check_if_can_start_ordering(timestamp);
                }
            }
        };

        let exchange_trades = match exchange {
            Bitmex => &mut self.bitmex_trades,
            Deribit => &mut self.deribit_trades,
        };

        for trade in trades {
            exchange_trades.push_back(trade.clone());
        }
        exchange_trades
            .retain(|t| timestamp - t.timestamp <= Duration::seconds(FEATURES_DATA_LIFETIME));
    }

    fn handle_order_update(&mut self, ou: OrderUpdate) {
        self.pf.on_order_update(ou.clone());

        match ou.status {
            // Do nothing
            OrderStatus::Opened => {}

            // TODO: Persist
            OrderStatus::Filled => {}

            // Re-order if Bitmex order is cancelled
            OrderStatus::Cancelled if ou.exchange == Bitmex => {
                let ((latest_bitmex_bid_price, _), (latest_bitmex_ask_price, _)) =
                    self.bitmex_obs.ob.top().unwrap();
                let submit_price = if ou.quantity > 0 {
                    *latest_bitmex_ask_price - ORDER_PRICE_OFFSET
                } else {
                    *latest_bitmex_bid_price + ORDER_PRICE_OFFSET
                };
                match ou.exchange {
                    Bitmex => {
                        let new_ou = OrderUpdate {
                            status: OrderStatus::Submitted,
                            bid_price: *latest_bitmex_bid_price,
                            ask_price: *latest_bitmex_ask_price,
                            price: submit_price,
                            ..ou
                        };

                        self.to_exchange.send(Event::OrderUpdate(new_ou)).unwrap();
                    }
                    Deribit => unreachable!(),
                }
            }
            err => panic!("{:?}", err),
        }
    }

    /// Set `self.can_start_ordering` to `true` when all order book and trade events have occurred for sufficient amount of time based on `FEATURES_DATA_LIFETIME`
    fn check_if_can_start_ordering(&mut self, timestamp: DateTime<Utc>) {
        if let (Some(bobts), Some(dobts), Some(btts), Some(dtts)) = (
            self.first_bitmex_order_book_event_ts,
            self.first_deribit_order_book_event_ts,
            self.first_bitmex_trade_event_ts,
            self.first_deribit_trade_event_ts,
        ) {
            let ts_cutoff = timestamp - Duration::seconds(FEATURES_DATA_LIFETIME);
            if ts_cutoff > bobts && ts_cutoff > dobts && ts_cutoff > btts && ts_cutoff > dtts {
                self.can_start_ordering = true;
            }
        }
    }
}
