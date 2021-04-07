use crate::{
    constants::{FEATURES_DATA_LIFETIME, ORDER_BOOK_SNAPSHOT_INTERVAL},
    event::{OrderBookEventType, OrderBookUpdate},
    order_book::OrderBook,
    price::Price,
};
use chrono::{DateTime, Duration, Utc};
use std::collections::VecDeque;

#[derive(Debug)]
pub struct OrderBookSnapshots {
    pub ob: OrderBook,
    pub snapshots: VecDeque<(Vec<(Price, i32)>, Vec<(Price, i32)>, DateTime<Utc>)>, // Order book snapshots for the past 60 seconds
}

impl OrderBookSnapshots {
    pub fn new() -> Self {
        Self {
            ob: OrderBook::new(),
            snapshots: VecDeque::new(),
        }
    }

    pub fn update(
        &mut self,
        type_: OrderBookEventType,
        bids: Vec<OrderBookUpdate>,
        asks: Vec<OrderBookUpdate>,
        timestamp: DateTime<Utc>,
    ) {
        self.ob.update(type_, bids, asks, timestamp);

        // Get indexes of expired order book snapshots
        let mut unexpired_ss_index = 0;
        for (i, ss) in self.snapshots.iter().enumerate() {
            if timestamp - ss.2 <= Duration::seconds(FEATURES_DATA_LIFETIME) {
                unexpired_ss_index = i;
                break;
            }
        }
        self.snapshots.push_back(self.ob.snapshot());

        // drain(..) removes indexes from 0 to unexpired_ss_index (exclusive)
        // the if block is to include the latest snapshot that is not within range for accurate
        // calculation in the make_snapshot function
        self.snapshots.drain(
            0..if unexpired_ss_index > 0 {
                unexpired_ss_index - 1
            } else {
                0
            },
        );
    }

    /// 60s of 100ms time interval snapshots of 25 price levels
    pub fn make_snapshot(
        &self,
        timestamp: DateTime<Utc>,
    ) -> VecDeque<(Vec<(Price, i32)>, Vec<(Price, i32)>, DateTime<Utc>)> {
        let mut snapshots = VecDeque::new();
        let mut last_ss_ts: Option<DateTime<Utc>> = None;

        for i in 0..FEATURES_DATA_LIFETIME * 1000 / ORDER_BOOK_SNAPSHOT_INTERVAL {
            let ts_cutoff = timestamp - Duration::milliseconds(i * ORDER_BOOK_SNAPSHOT_INTERVAL);

            for snapshot in self.snapshots.iter().rev() {
                let ss_ts = snapshot.2;
                if (last_ss_ts == None || last_ss_ts.unwrap() > ts_cutoff) && ss_ts <= ts_cutoff {
                    snapshots.push_front(snapshot.clone());
                    break;
                }
                last_ss_ts = Some(snapshot.2);
            }
        }

        snapshots
    }
}
