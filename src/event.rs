use crate::{algo::Signal, order::OrderUpdate, price::Price, Exchange};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::cmp::max;

#[derive(Serialize, Deserialize, Eq, PartialEq, Clone, Debug)]
pub enum Event {
    Tick {
        exchange: Exchange,
        bid_price: Price,
        ask_price: Price,
        timestamp: DateTime<Utc>,
    },
    Market {
        exchange: Exchange,
        bid_price: Price,
        ask_price: Price,
        timestamp: DateTime<Utc>,
    },
    OrderBook {
        exchange: Exchange,
        type_: OrderBookEventType,
        bids: Vec<OrderBookUpdate>,
        asks: Vec<OrderBookUpdate>,
        timestamp: DateTime<Utc>,
    },
    Trade {
        exchange: Exchange,
        trades: Vec<Trade>,
        timestamp: DateTime<Utc>,
    },
    OrderUpdate(OrderUpdate),
    Signal(Signal),
}

impl Event {
    pub fn timestamp(&self) -> Option<&DateTime<Utc>> {
        match self {
            Event::Tick {
                exchange: _,
                bid_price: _,
                ask_price: _,
                timestamp,
            } => Some(timestamp),
            Event::Market {
                exchange: _,
                bid_price: _,
                ask_price: _,
                timestamp,
            } => Some(timestamp),
            Event::OrderBook {
                exchange: _,
                type_: _,
                bids: _,
                asks: _,
                timestamp,
            } => Some(timestamp),
            Event::Trade {
                exchange: _,
                trades: _,
                timestamp,
            } => Some(timestamp),
            Event::OrderUpdate(ou) => Some(&ou.timestamp),
            Event::Signal(signal) => match signal {
                Signal::None => None,
                Signal::NewTrade(bou, dou) => Some(max(&bou.timestamp, &dou.timestamp)),
                Signal::PriceAdjustment(ou) => Some(&ou.timestamp),
            },
        }
    }
}

#[derive(Serialize, Deserialize, Eq, PartialEq, Copy, Clone, Debug)]
pub enum OrderBookEventType {
    Snapshot,
    Update,
}

#[derive(Serialize, Deserialize, Eq, PartialEq, Clone, Debug)]
pub struct OrderBookUpdate {
    pub type_: OrderBookUpdateType,
    pub price: Price,
    pub size: Option<i32>,
}

#[derive(Serialize, Deserialize, Eq, PartialEq, Clone, Debug)]
pub enum OrderBookUpdateType {
    Update,
    Delete,
}

#[derive(Serialize, Deserialize, Eq, PartialEq, Clone, Debug)]
pub struct Trade {
    pub tick_direction: TickDirection,
    pub size: i32,
    pub price: Price,
    pub timestamp: DateTime<Utc>,
}

#[derive(Serialize, Deserialize, Eq, PartialEq, Clone, Debug)]
pub enum TickDirection {
    Plus,
    ZeroPlus,
    Minus,
    ZeroMinus,
}
