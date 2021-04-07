use chrono::{serde::ts_milliseconds, DateTime, Utc};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BitmexQuoteJson {
    pub data: Vec<BitmexQuoteJsonDatum>,
    pub action: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BitmexQuoteJsonDatum {
    pub ask_size: i32,
    pub bid_size: i32,
    pub ask_price: f32,
    pub bid_price: f32,
    pub timestamp: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BitmexOrderBookJson {
    pub data: Vec<BitmexOrderBookJsonDatum>,
    pub action: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BitmexOrderBookJsonDatum {
    pub id: i64,
    pub side: String,
    pub size: Option<i32>,
    pub price: Option<f32>,
}

#[derive(Default, Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BitmexTradeJson {
    pub table: String,
    pub action: String,
    pub data: Vec<BitmexTradeJsonDatum>,
}

#[derive(Default, Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BitmexTradeJsonDatum {
    pub timestamp: String,
    pub side: String,
    pub size: i32,
    pub price: f32,
    pub tick_direction: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DeribitQuoteJson {
    pub params: DeribitQuoteJsonParams,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DeribitQuoteJsonParams {
    pub data: DeribitQuoteJsonParamsData,
    pub channel: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DeribitQuoteJsonParamsData {
    #[serde(with = "ts_milliseconds")]
    pub timestamp: DateTime<Utc>,
    #[serde(rename = "best_ask_price")]
    pub best_ask_price: f32,
    #[serde(rename = "best_bid_price")]
    pub best_bid_price: f32,
    #[serde(rename = "best_ask_amount")]
    pub best_ask_amount: f32,
    #[serde(rename = "best_bid_amount")]
    pub best_bid_amount: f32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DeribitOrderBookJson {
    pub params: DeribitOrderBookJsonParams,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DeribitOrderBookJsonParams {
    pub data: DeribitOrderBookJsonParamsData,
    pub channel: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DeribitOrderBookJsonParamsData {
    pub asks: Vec<(String, f32, f32)>,
    pub bids: Vec<(String, f32, f32)>,
    pub type_: String,
    #[serde(with = "ts_milliseconds")]
    pub timestamp: DateTime<Utc>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DeribitTradeJson {
    pub params: DeribitTradeJsonParams,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DeribitTradeJsonParams {
    pub channel: String,
    pub data: Vec<DeribitTradeJsonDatum>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DeribitTradeJsonDatum {
    #[serde(with = "ts_milliseconds")]
    pub timestamp: DateTime<Utc>,
    #[serde(rename = "tick_direction")]
    pub tick_direction: i32,
    pub price: f32,
    pub direction: String,
    pub amount: f32,
}
