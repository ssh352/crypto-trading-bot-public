use crate::{price::Price, Exchange};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Eq, PartialEq, Clone, Debug)]
pub struct Order {
    pub exchange: Exchange,
    pub id: String,
    pub quantity: i32,
    pub submit_price: Price,
    pub submit_timestamp: DateTime<Utc>,
    pub updates: Vec<OrderUpdate>,
}

impl Order {
    pub fn status(&self) -> &OrderStatus {
        let last_ou = self.updates.last().unwrap();
        &last_ou.status
    }
}

#[derive(Serialize, Deserialize, Eq, PartialEq, Clone, Debug)]
pub struct OrderUpdate {
    pub exchange: Exchange,
    pub id: String,
    pub status: OrderStatus,
    pub quantity: i32,
    pub bid_price: Price,
    pub ask_price: Price,
    pub price: Price,
    pub timestamp: DateTime<Utc>,
}

#[derive(Serialize, Deserialize, Eq, PartialEq, Copy, Clone, Debug)]
pub enum OrderStatus {
    Submitted,
    Received,
    Opened,
    Filled,
    Cancelled,
    FailedToFill,
}
