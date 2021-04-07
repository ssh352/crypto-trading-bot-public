use crate::{
    constants::ORDER_BOOK_SNAPSHOT_LEVELS,
    event::{OrderBookEventType, OrderBookUpdate, OrderBookUpdateType},
    price::Price,
};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct OrderBook {
    bids: BTreeMap<Price, i32>,
    asks: BTreeMap<Price, i32>,
    timestamp: Option<DateTime<Utc>>,
}

impl OrderBook {
    pub fn new() -> Self {
        Self {
            bids: BTreeMap::new(),
            asks: BTreeMap::new(),
            timestamp: None,
        }
    }

    pub fn update(
        &mut self,
        type_: OrderBookEventType,
        bids: Vec<OrderBookUpdate>,
        asks: Vec<OrderBookUpdate>,
        timestamp: DateTime<Utc>,
    ) {
        match type_ {
            OrderBookEventType::Snapshot => {
                self.bids = BTreeMap::new();
                self.asks = BTreeMap::new();
            }
            OrderBookEventType::Update => {}
        }
        for bid in bids {
            match (&bid.type_, bid.size) {
                (OrderBookUpdateType::Update, Some(size)) => {
                    self.bids.insert(bid.price, size);
                }
                (OrderBookUpdateType::Delete, None) => {
                    self.bids.remove(&bid.price);
                }
                err => panic!("{:?}", err),
            }
        }
        for ask in asks {
            match (&ask.type_, ask.size) {
                (OrderBookUpdateType::Update, Some(size)) => {
                    self.asks.insert(ask.price, size);
                }
                (OrderBookUpdateType::Delete, None) => {
                    self.asks.remove(&ask.price);
                }
                err => panic!("{:?}", err),
            }
        }
        self.timestamp = Some(timestamp);
    }

    pub fn top(&self) -> Option<((&Price, &i32), (&Price, &i32))> {
        match (self.bids.last_key_value(), self.asks.first_key_value()) {
            (Some(bid), Some(ask)) => Some((bid, ask)),
            _ => None,
        }
    }

    /// First indices of bids and asks are closest to mid price
    pub fn snapshot(&self) -> (Vec<(Price, i32)>, Vec<(Price, i32)>, DateTime<Utc>) {
        let mut bids = Vec::new();
        for (i, (price, qty)) in self.bids.iter().rev().enumerate() {
            bids.push((*price, *qty));
            if i == ORDER_BOOK_SNAPSHOT_LEVELS - 1 {
                break;
            }
        }

        let mut asks = Vec::new();
        for (i, (price, qty)) in self.asks.iter().enumerate() {
            asks.push((*price, *qty));
            if i == ORDER_BOOK_SNAPSHOT_LEVELS - 1 {
                break;
            }
        }

        (bids, asks, self.timestamp.unwrap())
    }
}
