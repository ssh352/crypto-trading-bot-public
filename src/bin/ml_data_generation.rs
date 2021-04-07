#![feature(or_patterns)]

use anyhow::Result;
use chrono::{DateTime, Duration, NaiveDateTime, Utc};
use crypto_trading_bot::{
    algo::PairsTradingAlgorithm,
    constants::*,
    event::{Event, Trade},
    features::generate_features,
    order::{Order, OrderStatus, OrderUpdate},
    order_book_snapshots::OrderBookSnapshots,
    price::Price,
    Exchange::*,
};
use sqlx::{
    postgres::{PgPoolOptions, PgRow},
    Pool, Postgres, Row,
};
use std::collections::{HashMap, VecDeque};
use tokio::stream::StreamExt;

#[derive(Clone)]
struct PairTrade {
    bitmex_order: Order,
    deribit_order: Order,
    max_long_delta_before_first_fill: Option<f64>,
    min_short_delta_before_first_fill: Option<f64>,
    max_long_delta_after_first_fill_and_before_second_fill: Option<f64>,
    min_short_delta_after_first_fill_and_before_second_fill: Option<f64>,
}

struct MlDataGeneration {
    pool: Pool<Postgres>,

    bitmex_order_id_to_pair_trade_id_map: HashMap<String, String>,
    deribit_order_id_to_pair_trade_id_map: HashMap<String, String>,
    unfilled_pair_trades: HashMap<String, PairTrade>,

    last_long_trade_ts: DateTime<Utc>,
    last_short_trade_ts: DateTime<Utc>,
    first_bitmex_order_book_event_ts: Option<DateTime<Utc>>,
    first_deribit_order_book_event_ts: Option<DateTime<Utc>>,
    first_bitmex_trade_event_ts: Option<DateTime<Utc>>,
    first_deribit_trade_event_ts: Option<DateTime<Utc>>,
    can_start_ordering: bool,

    latest_tick_prices: ((Price, Price), (Price, Price)),
    latest_market_prices: ((Price, Price), (Price, Price)),

    // Features
    bitmex_prices: VecDeque<(Price, Price, DateTime<Utc>)>,
    deribit_prices: VecDeque<(Price, Price, DateTime<Utc>)>,
    bitmex_obs: OrderBookSnapshots,
    deribit_obs: OrderBookSnapshots,
    bitmex_trades: VecDeque<Trade>,
    deribit_trades: VecDeque<Trade>,
}

impl MlDataGeneration {
    async fn new() -> Self {
        Self {
            pool: PgPoolOptions::new()
                .max_connections(5)
                .connect(DB_URL)
                .await
                .unwrap(),

            bitmex_order_id_to_pair_trade_id_map: HashMap::new(),
            deribit_order_id_to_pair_trade_id_map: HashMap::new(),
            unfilled_pair_trades: HashMap::new(),

            last_long_trade_ts: DateTime::<Utc>::from_utc(NaiveDateTime::from_timestamp(0, 0), Utc),
            last_short_trade_ts: DateTime::<Utc>::from_utc(
                NaiveDateTime::from_timestamp(0, 0),
                Utc,
            ),
            first_bitmex_order_book_event_ts: None,
            first_deribit_order_book_event_ts: None,
            first_bitmex_trade_event_ts: None,
            first_deribit_trade_event_ts: None,
            can_start_ordering: false,

            latest_tick_prices: (
                (Price::from_f32(0.), Price::from_f32(0.)),
                (Price::from_f32(0.), Price::from_f32(0.)),
            ),
            latest_market_prices: (
                (Price::from_f32(0.), Price::from_f32(0.)),
                (Price::from_f32(0.), Price::from_f32(0.)),
            ),

            bitmex_prices: VecDeque::new(),
            deribit_prices: VecDeque::new(),
            bitmex_obs: OrderBookSnapshots::new(),
            deribit_obs: OrderBookSnapshots::new(),
            bitmex_trades: VecDeque::new(),
            deribit_trades: VecDeque::new(),
        }
    }

    async fn run(mut self) -> Result<()> {
        let cursor_pool = self.pool.clone();

        let mut cursor = sqlx::query("SELECT json, timestamp FROM events ORDER BY timestamp ASC")
            .map(|row: PgRow| serde_json::from_value::<Event>(row.get(0)).unwrap())
            .fetch(&cursor_pool);

        while let Some(event) = cursor.try_next().await? {
            match event {
                Event::Tick { .. } => self.handle_tick_event(event).await,
                Event::Market { .. } => self.handle_market_event(event).await,
                Event::OrderBook { .. } => self.handle_order_book_event(event).await,
                Event::Trade { .. } => self.handle_trade_event(event),
                err => panic!("{:?}", err),
            }
        }

        Ok(())
    }

    async fn handle_tick_event(&mut self, tick_event: Event) {
        match tick_event {
            Event::Tick {
                exchange,
                bid_price,
                ask_price,
                timestamp,
            } => {
                // Update tick prices
                match exchange {
                    Bitmex => self.latest_tick_prices.0 = (bid_price, ask_price),
                    Deribit => self.latest_tick_prices.1 = (bid_price, ask_price),
                }

                // Update recent prices
                let exchange_prices = match exchange {
                    Bitmex => &mut self.bitmex_prices,
                    Deribit => &mut self.deribit_prices,
                };
                exchange_prices.push_back((bid_price, ask_price, timestamp));
                exchange_prices
                    .retain(|p| timestamp - p.2 <= Duration::seconds(FEATURES_DATA_LIFETIME));

                let mut filled_order_updates_and_their_pair_trades = vec![];
                let mut filled_order_ids = vec![];

                for unfilled_pair_trade_and_deltas in &mut self.unfilled_pair_trades.values_mut() {
                    // Calculate min/max deltas reached
                    // TODO:
                    // 1. Both unfilled
                    // 2. Bitmex filled, Deribit unfilled
                    // 3. Deribit filled, Bitmex unfilled
                    match (
                        unfilled_pair_trade_and_deltas.bitmex_order.updates.last(),
                        unfilled_pair_trade_and_deltas.deribit_order.updates.last(),
                    ) {
                        (Some(latest_bitmex_order_update), Some(latest_deribit_order_update)) => {
                            let portfolio_is_buy =
                                unfilled_pair_trade_and_deltas.bitmex_order.quantity > 0;

                            match (
                                latest_bitmex_order_update.status,
                                latest_deribit_order_update.status,
                            ) {
                                (
                                    OrderStatus::Cancelled
                                    | OrderStatus::Submitted
                                    | OrderStatus::Opened,
                                    OrderStatus::Submitted | OrderStatus::Opened,
                                ) => {
                                    if portfolio_is_buy {
                                        // max_long_delta_before_first_fill
                                        let long_delta = PairsTradingAlgorithm::long_delta(
                                            self.latest_tick_prices.0 .1,
                                            self.latest_tick_prices.1 .0,
                                        );
                                        match unfilled_pair_trade_and_deltas
                                            .max_long_delta_before_first_fill
                                        {
                                            Some(max_long_delta_before_first_fill) => {
                                                unfilled_pair_trade_and_deltas
                                                    .max_long_delta_before_first_fill = Some(
                                                    long_delta
                                                        .max(max_long_delta_before_first_fill),
                                                );
                                            }
                                            err => panic!("{:?}", err),
                                        };
                                    } else {
                                        // min_short_delta_before_first_fill
                                        let short_delta = PairsTradingAlgorithm::short_delta(
                                            self.latest_tick_prices.0 .0,
                                            self.latest_tick_prices.1 .1,
                                        );
                                        match unfilled_pair_trade_and_deltas
                                            .min_short_delta_before_first_fill
                                        {
                                            Some(min_short_delta_before_first_fill) => {
                                                unfilled_pair_trade_and_deltas
                                                    .min_short_delta_before_first_fill = Some(
                                                    short_delta
                                                        .min(min_short_delta_before_first_fill),
                                                );
                                            }
                                            err => panic!("{:?}", err),
                                        };
                                    }
                                }
                                (
                                    OrderStatus::Filled,
                                    OrderStatus::Submitted | OrderStatus::Opened,
                                ) => {
                                    if portfolio_is_buy {
                                        // max_long_delta_after_first_fill_and_before_second_fill
                                        let long_delta = PairsTradingAlgorithm::long_delta(
                                            latest_bitmex_order_update.ask_price, // Bitmex filled
                                            self.latest_tick_prices.1 .0,
                                        );
                                        match unfilled_pair_trade_and_deltas
                                            .max_long_delta_after_first_fill_and_before_second_fill
                                        {
                                            Some(
                                                max_long_delta_after_first_fill_and_before_second_fill,
                                            ) => {
                                                unfilled_pair_trade_and_deltas
                                                    .max_long_delta_after_first_fill_and_before_second_fill = Some(
                                                    long_delta
                                                        .max(max_long_delta_after_first_fill_and_before_second_fill),
                                                );
                                            }
                                            None => {
                                                unfilled_pair_trade_and_deltas
                                                .max_long_delta_after_first_fill_and_before_second_fill = Some(
                                                long_delta
                                            );
                                            }
                                        };
                                    } else {
                                        // min_short_delta_after_first_fill_and_before_second_fill
                                        let short_delta = PairsTradingAlgorithm::short_delta(
                                            latest_bitmex_order_update.bid_price, // Bitmex filled
                                            self.latest_tick_prices.1 .1,
                                        );
                                        match unfilled_pair_trade_and_deltas
                                            .min_short_delta_after_first_fill_and_before_second_fill
                                        {
                                            Some(
                                                min_short_delta_after_first_fill_and_before_second_fill,
                                            ) => {
                                                unfilled_pair_trade_and_deltas
                                                    .min_short_delta_after_first_fill_and_before_second_fill = Some(
                                                    short_delta
                                                        .min(min_short_delta_after_first_fill_and_before_second_fill),
                                                );
                                            }
                                            None => {
                                                unfilled_pair_trade_and_deltas
                                                .min_short_delta_after_first_fill_and_before_second_fill = Some(
                                                short_delta
                                            );
                                            }
                                        };
                                    }
                                }
                                (
                                    OrderStatus::Cancelled
                                    | OrderStatus::Submitted
                                    | OrderStatus::Opened,
                                    OrderStatus::Filled,
                                ) => {
                                    if portfolio_is_buy {
                                        // max_long_delta_after_first_fill_and_before_second_fill
                                        let long_delta = PairsTradingAlgorithm::long_delta(
                                            self.latest_tick_prices.0 .1,
                                            latest_deribit_order_update.bid_price, // Deribit filled
                                        );
                                        match unfilled_pair_trade_and_deltas
                                            .max_long_delta_after_first_fill_and_before_second_fill
                                        {
                                            Some(
                                                max_long_delta_after_first_fill_and_before_second_fill,
                                            ) => {
                                                unfilled_pair_trade_and_deltas
                                                    .max_long_delta_after_first_fill_and_before_second_fill = Some(
                                                    long_delta
                                                        .max(max_long_delta_after_first_fill_and_before_second_fill),
                                                );
                                            }
                                            None => {
                                                unfilled_pair_trade_and_deltas
                                                .max_long_delta_after_first_fill_and_before_second_fill = Some(
                                                long_delta
                                            );
                                            }
                                        };
                                    } else {
                                        // min_short_delta_after_first_fill_and_before_second_fill
                                        let short_delta = PairsTradingAlgorithm::short_delta(
                                            self.latest_tick_prices.0 .0,
                                            latest_deribit_order_update.ask_price, // Deribit filled
                                        );
                                        match unfilled_pair_trade_and_deltas
                                            .min_short_delta_after_first_fill_and_before_second_fill
                                        {
                                            Some(
                                                min_short_delta_after_first_fill_and_before_second_fill,
                                            ) => {
                                                unfilled_pair_trade_and_deltas
                                                    .min_short_delta_after_first_fill_and_before_second_fill = Some(
                                                    short_delta
                                                        .min(min_short_delta_after_first_fill_and_before_second_fill),
                                                );
                                            }
                                            None => {
                                                unfilled_pair_trade_and_deltas
                                                .min_short_delta_after_first_fill_and_before_second_fill = Some(
                                                short_delta
                                            );
                                            }
                                        };
                                    }
                                }
                                err => panic!("{:?}", err),
                            }
                        }
                        err => panic!("{:?}", err),
                    }

                    // Handle Bitmex cancelled submissions and orders filling and persistence
                    let opt_unfilled_order = match exchange {
                        Bitmex => {
                            let bitmex_order = &mut unfilled_pair_trade_and_deltas.bitmex_order;
                            match bitmex_order.updates.last().unwrap().status {
                                OrderStatus::Filled => None,
                                _ => Some(bitmex_order),
                            }
                        }
                        Deribit => {
                            let deribit_order = &mut unfilled_pair_trade_and_deltas.deribit_order;
                            match deribit_order.updates.last().unwrap().status {
                                OrderStatus::Filled => None,
                                _ => Some(deribit_order),
                            }
                        }
                    };

                    if let Some(unfilled_order) = opt_unfilled_order {
                        let order_is_buy = unfilled_order.quantity > 0;

                        match unfilled_order.status() {
                            OrderStatus::Submitted => {
                                let submitted_ou = unfilled_order.updates.last().unwrap().clone();
                                let recv_ts = unfilled_order.submit_timestamp
                                    + Duration::milliseconds(LATENCY);

                                if recv_ts <= timestamp {
                                    let submit_price_did_not_touch = order_is_buy
                                        && unfilled_order.submit_price < ask_price
                                        || !order_is_buy && unfilled_order.submit_price > bid_price;

                                    let new_ou = if submit_price_did_not_touch {
                                        OrderUpdate {
                                            status: OrderStatus::Opened,
                                            bid_price,
                                            ask_price,
                                            timestamp,
                                            ..submitted_ou
                                        }
                                    } else {
                                        match exchange {
                                            Bitmex => OrderUpdate {
                                                status: OrderStatus::Cancelled,
                                                bid_price,
                                                ask_price,
                                                timestamp,
                                                ..submitted_ou
                                            },
                                            Deribit => OrderUpdate {
                                                status: OrderStatus::Opened,
                                                bid_price,
                                                ask_price,
                                                price: if order_is_buy {
                                                    ask_price - ORDER_PRICE_OFFSET
                                                } else {
                                                    bid_price + ORDER_PRICE_OFFSET
                                                },
                                                timestamp,
                                                ..submitted_ou
                                            },
                                        }
                                    };

                                    unfilled_order.updates.push(new_ou.clone());
                                }
                            }
                            OrderStatus::Opened => {
                                let opened_ou = unfilled_order.updates.last().unwrap().clone();

                                let is_filled = order_is_buy && opened_ou.price >= ask_price
                                    || !order_is_buy && opened_ou.price <= bid_price;

                                if is_filled {
                                    let filled_ou = OrderUpdate {
                                        status: OrderStatus::Filled,
                                        bid_price,
                                        ask_price,
                                        timestamp,
                                        ..opened_ou
                                    };

                                    unfilled_order.updates.push(filled_ou.clone());
                                    filled_order_ids.push(unfilled_order.id.clone());
                                    filled_order_updates_and_their_pair_trades
                                        .push((filled_ou, unfilled_pair_trade_and_deltas.clone()));
                                }
                            }
                            OrderStatus::Cancelled => {
                                // TODO: Handle cancelled Bitmex submissions
                            }
                            err => panic!("{:?}", err),
                        }
                    }
                }

                // Persist
                for (filled_order_update, pair_trade) in filled_order_updates_and_their_pair_trades
                {
                    self.persist_fill_order_update(filled_order_update, pair_trade)
                        .await;
                }

                // Remove filled ids and pair trades
                self.bitmex_order_id_to_pair_trade_id_map
                    .retain(|bitmex_order_id, _| !filled_order_ids.contains(&bitmex_order_id));
                self.deribit_order_id_to_pair_trade_id_map
                    .retain(|deribit_order_id, _| !filled_order_ids.contains(&deribit_order_id));
                self.unfilled_pair_trades.retain(|_, pair_trade| {
                    pair_trade.bitmex_order.updates.last().unwrap().status != OrderStatus::Filled
                        || pair_trade.deribit_order.updates.last().unwrap().status
                            != OrderStatus::Filled
                });

                print!(
                    "\r[{}] {} {} {}              ",
                    timestamp,
                    self.bitmex_order_id_to_pair_trade_id_map.len(),
                    self.deribit_order_id_to_pair_trade_id_map.len(),
                    self.unfilled_pair_trades.len()
                );
            }
            err => panic!("{:?}", err),
        }
    }
    async fn handle_market_event(&mut self, market_event: Event) {
        match market_event {
            Event::Market {
                exchange,
                bid_price,
                ask_price,
                timestamp,
            } => {
                // Update prices
                match exchange {
                    Bitmex => self.latest_market_prices.0 = (bid_price, ask_price),
                    Deribit => self.latest_market_prices.1 = (bid_price, ask_price),
                }

                if self.can_start_ordering {
                    let (
                        (bitmex_bid_price, bitmex_ask_price),
                        (deribit_bid_price, deribit_ask_price),
                    ) = self.latest_market_prices;

                    let long_delta =
                        PairsTradingAlgorithm::long_delta(bitmex_ask_price, deribit_bid_price);
                    let short_delta =
                        PairsTradingAlgorithm::short_delta(bitmex_bid_price, deribit_ask_price);

                    // Should long portfolio
                    if self.last_long_trade_ts < timestamp - Duration::seconds(ORDER_INTERVAL)
                        && long_delta < -DELTA_THRESHOLD
                    {
                        // Create orders
                        let bitmex_qty = MAX_POS_SIZE * 2;
                        let bitmex_order_price = bitmex_ask_price - ORDER_PRICE_OFFSET;
                        let bitmex_id = format!(
                            "[{:?}] {} {} * {} @ delta: {:?}",
                            timestamp, Bitmex, bitmex_qty, bitmex_order_price, long_delta
                        );

                        let deribit_qty = -bitmex_qty;
                        let deribit_order_price = deribit_bid_price + ORDER_PRICE_OFFSET;
                        let deribit_id = format!(
                            "[{:?}] {} {} * {} @ delta: {:?}",
                            timestamp, Deribit, deribit_qty, deribit_order_price, long_delta
                        );

                        let submit_timestamp = timestamp + Duration::milliseconds(LATENCY);

                        let bitmex_order = Order {
                            exchange: Bitmex,
                            id: bitmex_id.clone(),
                            quantity: bitmex_qty,
                            submit_price: bitmex_order_price,
                            submit_timestamp, // Add latency to account for time it takes for market event to reach client
                            updates: vec![OrderUpdate {
                                exchange: Bitmex,
                                id: bitmex_id,
                                status: OrderStatus::Submitted,
                                quantity: bitmex_qty,
                                bid_price: bitmex_bid_price,
                                ask_price: bitmex_ask_price,
                                price: bitmex_order_price,
                                timestamp: submit_timestamp,
                            }],
                        };

                        let deribit_order = Order {
                            exchange: Deribit,
                            id: deribit_id.clone(),
                            quantity: deribit_qty,
                            submit_price: deribit_order_price,
                            submit_timestamp,
                            updates: vec![OrderUpdate {
                                exchange: Deribit,
                                id: deribit_id,
                                status: OrderStatus::Submitted,
                                quantity: deribit_qty,
                                bid_price: deribit_bid_price,
                                ask_price: deribit_ask_price,
                                price: deribit_order_price,
                                timestamp: submit_timestamp,
                            }],
                        };

                        self.persist_new_pair_trade_submission(&bitmex_order, &deribit_order)
                            .await;

                        // Persist
                        let pair_trade_id =
                            format!("{} | {}", bitmex_order.id.clone(), deribit_order.id.clone());
                        self.bitmex_order_id_to_pair_trade_id_map
                            .insert(bitmex_order.id.clone(), pair_trade_id.clone());
                        self.deribit_order_id_to_pair_trade_id_map
                            .insert(deribit_order.id.clone(), pair_trade_id.clone());
                        self.unfilled_pair_trades.insert(
                            pair_trade_id,
                            PairTrade {
                                bitmex_order,
                                deribit_order,
                                max_long_delta_before_first_fill: Some(long_delta),
                                min_short_delta_before_first_fill: None,
                                max_long_delta_after_first_fill_and_before_second_fill: None,
                                min_short_delta_after_first_fill_and_before_second_fill: None,
                            },
                        );

                        self.last_long_trade_ts = timestamp;
                    }

                    // Should short portfolio
                    if self.last_short_trade_ts < timestamp - Duration::seconds(ORDER_INTERVAL)
                        && short_delta > DELTA_THRESHOLD
                    {
                        // Create orders
                        let bitmex_qty = -MAX_POS_SIZE * 2;
                        let bitmex_order_price = bitmex_bid_price + ORDER_PRICE_OFFSET;
                        let bitmex_id = format!(
                            "[{:?}] {} {} * {} @ delta: {:?}",
                            timestamp, Bitmex, bitmex_qty, bitmex_order_price, short_delta
                        );

                        let deribit_qty = -bitmex_qty;
                        let deribit_order_price = deribit_ask_price - ORDER_PRICE_OFFSET;
                        let deribit_id = format!(
                            "[{:?}] {} {} * {} @ delta: {:?}",
                            timestamp, Deribit, deribit_qty, deribit_order_price, short_delta
                        );

                        let submit_timestamp = timestamp + Duration::milliseconds(LATENCY);

                        let bitmex_order = Order {
                            exchange: Bitmex,
                            id: bitmex_id.clone(),
                            quantity: bitmex_qty,
                            submit_price: bitmex_order_price,
                            submit_timestamp, // Add latency to account for time it takes for market event to reach client
                            updates: vec![OrderUpdate {
                                exchange: Bitmex,
                                id: bitmex_id,
                                status: OrderStatus::Submitted,
                                quantity: bitmex_qty,
                                bid_price: bitmex_bid_price,
                                ask_price: bitmex_ask_price,
                                price: bitmex_order_price,
                                timestamp: submit_timestamp,
                            }],
                        };

                        let deribit_order = Order {
                            exchange: Deribit,
                            id: deribit_id.clone(),
                            quantity: deribit_qty,
                            submit_price: deribit_order_price,
                            submit_timestamp,
                            updates: vec![OrderUpdate {
                                exchange: Deribit,
                                id: deribit_id,
                                status: OrderStatus::Submitted,
                                quantity: deribit_qty,
                                bid_price: deribit_bid_price,
                                ask_price: deribit_ask_price,
                                price: deribit_order_price,
                                timestamp: submit_timestamp,
                            }],
                        };

                        self.persist_new_pair_trade_submission(&bitmex_order, &deribit_order)
                            .await;

                        let pair_trade_id =
                            format!("{} | {}", bitmex_order.id.clone(), deribit_order.id.clone());
                        self.bitmex_order_id_to_pair_trade_id_map
                            .insert(bitmex_order.id.clone(), pair_trade_id.clone());
                        self.deribit_order_id_to_pair_trade_id_map
                            .insert(deribit_order.id.clone(), pair_trade_id.clone());
                        self.unfilled_pair_trades.insert(
                            pair_trade_id,
                            PairTrade {
                                bitmex_order,
                                deribit_order,
                                max_long_delta_before_first_fill: None,
                                min_short_delta_before_first_fill: Some(short_delta),
                                max_long_delta_after_first_fill_and_before_second_fill: None,
                                min_short_delta_after_first_fill_and_before_second_fill: None,
                            },
                        );

                        self.last_short_trade_ts = timestamp;
                    }
                }
            }
            err => panic!("{:?}", err),
        }
    }

    async fn handle_order_book_event(&mut self, order_book_event: Event) {
        match order_book_event {
            Event::OrderBook {
                exchange,
                type_,
                bids,
                asks,
                timestamp,
            } => {
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
                        })
                        .await;
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
            err => panic!("{:?}", err),
        }
    }

    fn handle_trade_event(&mut self, trade_event: Event) {
        match trade_event {
            Event::Trade {
                exchange,
                trades,
                timestamp,
            } => {
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
                exchange_trades.retain(|t| {
                    timestamp - t.timestamp <= Duration::seconds(FEATURES_DATA_LIFETIME)
                });
            }
            err => panic!("{:?}", err),
        }
    }

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

    async fn persist_fill_order_update(
        &mut self,
        order_update: OrderUpdate,
        pair_trade: PairTrade,
    ) {
        // deltas in PairTrade
        let (is_filled, d1, d2, d3, d4) = (
            pair_trade.bitmex_order.updates.last().unwrap().status == OrderStatus::Filled
                && pair_trade.deribit_order.updates.last().unwrap().status == OrderStatus::Filled,
            pair_trade.max_long_delta_before_first_fill,
            pair_trade.min_short_delta_before_first_fill,
            pair_trade.max_long_delta_after_first_fill_and_before_second_fill,
            pair_trade.min_short_delta_after_first_fill_and_before_second_fill,
        );

        match order_update.exchange {
            Bitmex => {
                sqlx::query!("UPDATE pair_trades_ml SET is_filled = $2, bitmex_fill_timestamp = $3, max_long_delta_before_first_fill = $4, min_short_delta_before_first_fill = $5, max_long_delta_after_first_fill_and_before_second_fill = $6, min_short_delta_after_first_fill_and_before_second_fill = $7 WHERE bitmex_order_id = $1", order_update.id, is_filled, order_update.timestamp, d1, d2, d3, d4 )
            .execute(&self.pool)
            .await
            .unwrap();
            }
            Deribit => {
                sqlx::query!("UPDATE pair_trades_ml SET is_filled = $2, deribit_fill_timestamp = $3, max_long_delta_before_first_fill = $4, min_short_delta_before_first_fill = $5, max_long_delta_after_first_fill_and_before_second_fill = $6, min_short_delta_after_first_fill_and_before_second_fill = $7 WHERE deribit_order_id = $1", order_update.id, is_filled, order_update.timestamp, d1, d2, d3, d4 )
            .execute(&self.pool)
            .await
            .unwrap();
            }
        };
    }

    async fn persist_new_pair_trade_submission(
        &mut self,
        bitmex_order: &Order,
        deribit_order: &Order,
    ) {
        // Otherwise will panic halfway through execution when there is not enough obs data
        if self.bitmex_obs.snapshots.is_empty()
            || self.deribit_obs.snapshots.is_empty()
            || self.bitmex_obs.snapshots.len() < 61
            || self.deribit_obs.snapshots.len() < 61
        {
            return;
        }

        let submit_timestamp = if bitmex_order.submit_timestamp == deribit_order.submit_timestamp {
            bitmex_order.submit_timestamp
        } else {
            panic!();
        };

        let quantity = bitmex_order.quantity;
        let recent_prices_json = serde_json::json!(vec![&self.bitmex_prices, &self.deribit_prices]);
        let order_book_snapshots_json = serde_json::json!(vec![
            self.bitmex_obs.make_snapshot(submit_timestamp),
            self.deribit_obs.make_snapshot(submit_timestamp)
        ]);
        let recent_trades_json = serde_json::json!(vec![&self.bitmex_trades, &self.deribit_trades]);
        let bitmex_features = generate_features(
            bitmex_order,
            &self.bitmex_obs.snapshots,
            &self.bitmex_trades,
        );
        let deribit_features = generate_features(
            deribit_order,
            &self.deribit_obs.snapshots,
            &self.deribit_trades,
        );
        let features = serde_json::json!(vec![bitmex_features, deribit_features]);
        sqlx::query!(
            "INSERT INTO pair_trades_ml (bitmex_order_id, deribit_order_id, quantity, recent_prices, order_book_snapshots, recent_trades, features, submit_timestamp) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)",
            bitmex_order.id, deribit_order.id, quantity, recent_prices_json, order_book_snapshots_json, recent_trades_json, features, submit_timestamp
        )
        .execute(&self.pool)
        .await
        .unwrap();
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    MlDataGeneration::new().await.run().await
}
