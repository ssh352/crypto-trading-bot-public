use crate::{
    constants::*,
    event::Event,
    order::{OrderStatus, OrderUpdate},
    portfolio::Portfolio,
    price::Price,
    Exchange::*,
};
use serde::{Deserialize, Serialize};
use std::cmp::max;

pub struct PairsTradingAlgorithm;

impl PairsTradingAlgorithm {
    pub fn new() -> Self {
        Self {}
    }

    /// - if portfolio empty, check if can long or short (by calculating stat arb delta)
    /// - else if portfolio is filled, check current portfolio position then check if can order in the other direction
    /// - else return no signal
    pub fn generate_signal(
        &self,
        pf: &Portfolio,
        latest_bitmex_market_event: &Event,
        latest_deribit_market_event: &Event,
    ) -> Signal {
        match (latest_bitmex_market_event, latest_deribit_market_event) {
            (
                Event::Market {
                    exchange: bitmex_exchange,
                    bid_price: bitmex_bid_price,
                    ask_price: bitmex_ask_price,
                    timestamp: bitmex_market_event_timestamp,
                },
                Event::Market {
                    exchange: deribit_exchange,
                    bid_price: deribit_bid_price,
                    ask_price: deribit_ask_price,
                    timestamp: deribit_market_event_timestamp,
                },
            ) if *bitmex_exchange == Bitmex && *deribit_exchange == Deribit => {
                let long_delta = Self::long_delta(*bitmex_ask_price, *deribit_bid_price);
                let short_delta = Self::short_delta(*bitmex_bid_price, *deribit_ask_price);
                let order_timestamp = *max(
                    bitmex_market_event_timestamp,
                    deribit_market_event_timestamp,
                );

                let (delta, order_qty_signed, bitmex_order_price, deribit_order_price) = if pf
                    .is_empty()
                    || pf.is_filled()
                        && pf.current_bitmex_filled_pos() == 0
                        && pf.current_deribit_filled_pos() == 0
                {
                    if long_delta < -DELTA_THRESHOLD {
                        (
                            long_delta,
                            MAX_POS_SIZE * 2,
                            *bitmex_ask_price - ORDER_PRICE_OFFSET,
                            *deribit_bid_price + ORDER_PRICE_OFFSET,
                        )
                    } else if short_delta > DELTA_THRESHOLD {
                        (
                            short_delta,
                            -MAX_POS_SIZE * 2,
                            *bitmex_bid_price + ORDER_PRICE_OFFSET,
                            *deribit_ask_price - ORDER_PRICE_OFFSET,
                        )
                    } else {
                        return Signal::None;
                    }
                } else if pf.is_filled() {
                    if pf.is_short() && long_delta < -DELTA_THRESHOLD {
                        (
                            long_delta,
                            MAX_POS_SIZE * 2,
                            *bitmex_ask_price - ORDER_PRICE_OFFSET,
                            *deribit_bid_price + ORDER_PRICE_OFFSET,
                        )
                    } else if pf.is_long() && short_delta > DELTA_THRESHOLD {
                        (
                            short_delta,
                            -MAX_POS_SIZE * 2,
                            *bitmex_bid_price + ORDER_PRICE_OFFSET,
                            *deribit_ask_price - ORDER_PRICE_OFFSET,
                        )
                    } else {
                        return Signal::None;
                    }
                } else {
                    return Signal::None;
                };
                return Signal::NewTrade(
                    OrderUpdate {
                        exchange: Bitmex,
                        id: format!(
                            "[{:?}] {} {} * {} @ delta: {:?}",
                            order_timestamp, Bitmex, order_qty_signed, bitmex_order_price, delta
                        ),
                        status: OrderStatus::Submitted,
                        quantity: order_qty_signed,
                        bid_price: *bitmex_bid_price,
                        ask_price: *bitmex_ask_price,
                        price: bitmex_order_price,
                        timestamp: order_timestamp,
                    },
                    OrderUpdate {
                        exchange: Deribit,
                        id: format!(
                            "[{:?}] {} {} * {} @ delta: {:?}",
                            order_timestamp, Deribit, -order_qty_signed, deribit_order_price, delta
                        ),
                        status: OrderStatus::Submitted,
                        quantity: -order_qty_signed,
                        bid_price: *deribit_bid_price,
                        ask_price: *deribit_ask_price,
                        price: deribit_order_price,
                        timestamp: order_timestamp,
                    },
                );
            }
            err => panic!("{:?}", err),
        }
    }

    pub fn long_delta(bitmex_ask_price: Price, deribit_bid_price: Price) -> f64 {
        let bitmex_order_price = bitmex_ask_price - ORDER_PRICE_OFFSET;
        let deribit_order_price = deribit_bid_price + ORDER_PRICE_OFFSET;
        bitmex_order_price.ln() - COINT_COEFF * deribit_order_price.ln() - EQUIL_VAL
    }

    pub fn short_delta(bitmex_bid_price: Price, deribit_ask_price: Price) -> f64 {
        let bitmex_order_price = bitmex_bid_price + ORDER_PRICE_OFFSET;
        let deribit_order_price = deribit_ask_price - ORDER_PRICE_OFFSET;
        bitmex_order_price.ln() - COINT_COEFF * deribit_order_price.ln() - EQUIL_VAL
    }
}

#[derive(Serialize, Deserialize, Eq, PartialEq, Clone, Debug)]
pub enum Signal {
    None,
    NewTrade(OrderUpdate, OrderUpdate),
    PriceAdjustment(OrderUpdate),
}
