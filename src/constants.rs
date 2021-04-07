use crate::price::Price;

pub const DB_URL: &str = "";
pub const LATENCY: i64 = 95; // milliseconds
pub const BITMEX_REBATE_RATE: f64 = 0.00025;

// Order
pub const MAX_POS_SIZE: i32 = 10;
pub const ORDER_PRICE_OFFSET: Price = Price { price_x100: 50 };
pub const COINT_COEFF: f64 = 1.00020118391358;
pub const EQUIL_VAL: f64 = -0.0026760984661962;
pub const DELTA_THRESHOLD: f64 = 0.0005;
pub const CLOSE_UNFILLED_DELTA_THRESHOLD: f64 = -99999.; // TODO: Some unrealistic number so it doesn't affect our market order testing, (which I don't think it will anyway but just in case)

// Fill probability ML model
pub const FEATURES_DATA_LIFETIME: i64 = 60; // seconds
pub const ORDER_BOOK_SNAPSHOT_LEVELS: usize = 25; // seconds
pub const ORDER_BOOK_SNAPSHOT_INTERVAL: i64 = 100; // milliseconds
pub const ORDER_INTERVAL: i64 = 3; // seconds
pub const FILL_TIME_WINDOW: i64 = 60; // seconds
