use crate::{
    algo::Signal,
    constants::*,
    event::Event,
    order::{Order, OrderStatus, OrderUpdate},
    price::Price,
    Exchange::{self, *},
};
use chrono::{DateTime, Duration, Utc};
use crossbeam::channel::{Receiver, Sender};
use sqlx::{
    postgres::{PgPoolOptions, PgRow},
    Row,
};
use tokio::{runtime::Runtime, stream::StreamExt};

pub struct SimulatedExchange {
    to_client: Sender<Event>,
    from_client: Receiver<Event>,

    // List of submitted orders that have yet to be filled
    unfilled_orders: Vec<Order>,

    order_updates_to_be_sent: Vec<OrderUpdate>,
}

impl SimulatedExchange {
    pub fn new(to_client: Sender<Event>, from_client: Receiver<Event>) -> Self {
        Self {
            to_client,
            from_client,
            unfilled_orders: vec![],
            order_updates_to_be_sent: vec![],
        }
    }

    pub fn run(mut self) {
        let mut rt = Runtime::new().unwrap();
        let pool = rt
            .block_on(PgPoolOptions::new().max_connections(5).connect(DB_URL))
            .unwrap();

        let mut cursor = sqlx::query("SELECT json, timestamp FROM events ORDER BY timestamp ASC")
            .map(|row: PgRow| serde_json::from_value::<Event>(row.get(0)).unwrap())
            .fetch(&pool);

        while let Some(Ok(event)) = rt.block_on(cursor.next()) {
            // Check if there are any order updates to be sent to client
            let mut sent_order_update_ids = vec![];
            for ou in &self.order_updates_to_be_sent {
                if let Some(ts) = event.timestamp() {
                    if ou.timestamp + Duration::milliseconds(LATENCY) <= *ts {
                        self.to_client.send(Event::OrderUpdate(ou.clone())).unwrap();
                        sent_order_update_ids.push(ou.id.clone());
                    }
                }
            }
            self.order_updates_to_be_sent
                .retain(|ou| !sent_order_update_ids.contains(&ou.id));

            match event {
                Event::Tick {
                    exchange,
                    bid_price,
                    ask_price,
                    timestamp,
                } => self.handle_tick_event(exchange, bid_price, ask_price, timestamp),

                Event::Market {
                    exchange,
                    bid_price: _,
                    ask_price: _,
                    timestamp: _,
                } => {
                    if exchange == Deribit {
                        self.to_client.send(event).unwrap();

                        let signal_event = self.from_client.recv().unwrap();
                        self.handle_signal_event(signal_event);
                    }
                }

                Event::OrderBook {
                    exchange,
                    type_: _,
                    bids: _,
                    asks: _,
                    timestamp: _,
                } => {
                    self.to_client.send(event).unwrap();

                    // If Bitmex order book event was sent to client, await a signal event reply
                    if exchange == Bitmex {
                        let signal_event = self.from_client.recv().unwrap();
                        self.handle_signal_event(signal_event);
                    }
                }

                _ => self.to_client.send(event).unwrap(),
            }
        }
    }

    fn handle_tick_event(
        &mut self,
        exchange: Exchange,
        bid_price: Price,
        ask_price: Price,
        timestamp: DateTime<Utc>,
    ) {
        let mut filled_orders = vec![];

        for order in &mut self.unfilled_orders {
            if order.exchange == exchange {
                let is_buy = order.quantity > 0;

                match order.status() {
                    OrderStatus::Submitted => {
                        let submitted_ou = order.updates.last().unwrap().clone();
                        let recv_ts = order.submit_timestamp + Duration::milliseconds(LATENCY);

                        if recv_ts <= timestamp {
                            let submit_price_did_not_touch = is_buy
                                && order.submit_price < ask_price
                                || !is_buy && order.submit_price > bid_price;

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
                                        price: if is_buy {
                                            ask_price - ORDER_PRICE_OFFSET
                                        } else {
                                            bid_price + ORDER_PRICE_OFFSET
                                        },
                                        timestamp,
                                        ..submitted_ou
                                    },
                                }
                            };

                            order.updates.push(new_ou.clone());

                            self.order_updates_to_be_sent.push(new_ou.clone());

                            // !
                            // self.to_client
                            //     .send(Event::OrderUpdate(new_ou.clone()))
                            //     .unwrap();

                            // if exchange == Bitmex && new_ou.status == OrderStatus::Cancelled {
                            //     let bm_reorder_ou_event = self.from_client.recv().unwrap();

                            //     match bm_reorder_ou_event {
                            //         Event::OrderUpdate(bm_reorder_ou) => {
                            //             order.updates.push(bm_reorder_ou);
                            //         }
                            //         err => panic!("{:?}", err),
                            //     }
                            // }
                        }
                    }
                    OrderStatus::Opened => {
                        let opened_ou = order.updates.last().unwrap().clone();

                        let is_filled = is_buy && opened_ou.price >= ask_price
                            || !is_buy && opened_ou.price <= bid_price;

                        if is_filled {
                            let filled_ou = OrderUpdate {
                                status: OrderStatus::Filled,
                                bid_price,
                                ask_price,
                                timestamp,
                                ..opened_ou
                            };

                            self.order_updates_to_be_sent.push(filled_ou.clone());

                            // !
                            // self.to_client.send(Event::OrderUpdate(filled_ou)).unwrap();

                            filled_orders.push(order.id.clone());
                        }
                    }
                    OrderStatus::Cancelled => {}
                    err => panic!("{:?}", err),
                }
            }
        }

        self.unfilled_orders
            .retain(|o| !filled_orders.contains(&o.id));
    }

    fn handle_signal_event(&mut self, signal_event: Event) {
        // Cases: new order, adjust price, cancel order

        match signal_event {
            Event::Signal(signal) => match signal {
                Signal::None => {}
                Signal::NewTrade(bou, dou) => {
                    self.unfilled_orders.push(Order {
                        exchange: Bitmex,
                        id: bou.id.clone(),
                        quantity: bou.quantity,
                        submit_price: bou.price,
                        submit_timestamp: bou.timestamp,
                        updates: vec![bou],
                    });
                    self.unfilled_orders.push(Order {
                        exchange: Deribit,
                        id: dou.id.clone(),
                        quantity: dou.quantity,
                        submit_price: dou.price,
                        submit_timestamp: dou.timestamp,
                        updates: vec![dou],
                    });
                }
                Signal::PriceAdjustment(ou) => {
                    // TODO: Fix this hacky auto-fill order and actually adjust the price etc.

                    self.unfilled_orders.retain(|o| o.id != ou.id);
                }
            },
            err => panic!("{:?}", err),
        }
    }
}
