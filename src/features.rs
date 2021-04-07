use crate::{
    constants::ORDER_BOOK_SNAPSHOT_INTERVAL,
    event::Trade,
    order::{Order, OrderUpdate},
    price::Price,
};
use chrono::{DateTime, Duration, Utc};
use serde::Serialize;
use serde_json::{json, Value};
use std::collections::{HashMap, VecDeque};

#[derive(Debug, Serialize)]
pub struct Feature {
    pub name: String,
    pub value: Value,
}

macro_rules! feature {
    ($e: expr) => {
        Feature {
            name: stringify!($e).to_string(),
            value: json!($e),
        }
    };
}

// "bid_ask_spread",
// "top_1_quantity_same_side",
// "top_1_quantity_opposite_side",
// "top_1_bid_ask_imbalance",
// "bid_imbalance_10_to_15",
// "ask_imbalance_10_to_15",
// "volume_order_imbalance_1000_ms_1_level",
// "trade_flow_imbalance_60s",
// "css_1000",
// "relative_strength_60_1",
// "ln_return_6000",
// "midprice_variance_60",
// "signed_trade_size_variance_60",
pub fn generate_chosen_features(
    submit_ou: &OrderUpdate,
    obs: &VecDeque<(Vec<(Price, i32)>, Vec<(Price, i32)>, DateTime<Utc>)>,
    rt: &VecDeque<Trade>,
) -> Vec<f64> {
    let is_buy = submit_ou.quantity > 0;
    let latest_obs = obs.back().unwrap();

    vec![
        (submit_ou.ask_price - submit_ou.bid_price).as_f64(),
        top_n_quantity(1, is_buy, true, latest_obs) as f64,
        top_n_quantity(1, is_buy, false, latest_obs) as f64,
        top_n_bid_ask_imbalance(1, latest_obs),
        bid_or_ask_imbalance(true, 10, 25, latest_obs),
        bid_or_ask_imbalance(false, 10, 25, latest_obs),
        volume_order_imbalance(1000, 1, obs) as f64,
        trade_flow_imbalance(60, rt, submit_ou.timestamp),
        css(1000, obs, submit_ou.timestamp),
        relative_strength_index(60, 1, rt, submit_ou.timestamp),
        ln_return(6000, obs),
        midprice_variance(60, obs, submit_ou.timestamp),
        signed_trade_size_variance(60, rt, submit_ou.timestamp),
    ]
}

/// Features inspired from the following papers:
/// - https://discovery.ucl.ac.uk/id/eprint/1359852/1/Chaiyakorn%20Yingsaeree%20-%20Thesis.pdf
/// - https://www.imperial.ac.uk/media/imperial-college/faculty-of-natural-sciences/department-of-mathematics/math-finance/Forecasting_cryptocurrency_prices.pdf
pub fn generate_features(
    order: &Order,
    obs: &VecDeque<(Vec<(Price, i32)>, Vec<(Price, i32)>, DateTime<Utc>)>, // order book snapshots for the past 60s for every 0.1s
    rt: &VecDeque<Trade>,                                                  // recent trades
) -> Vec<Feature> {
    let mut features = Vec::new();

    let submit_ou = order.updates.first().unwrap();
    let is_buy = order.quantity > 0;
    let latest_obs = obs.back().unwrap();

    //=== Features ===

    // Spread
    let bid_ask_spread = (submit_ou.ask_price - submit_ou.bid_price).as_f32();
    features.push(feature!(bid_ask_spread));
    let diff_same_side_price = diff_same_side_price(is_buy, submit_ou);
    features.push(feature!(diff_same_side_price));

    // Top n price level volume
    let top_1_quantity_same_side = top_n_quantity(1, is_buy, true, latest_obs);
    features.push(feature!(top_1_quantity_same_side));
    let top_5_quantity_same_side = top_n_quantity(5, is_buy, true, latest_obs);
    features.push(feature!(top_5_quantity_same_side));
    let top_10_quantity_same_side = top_n_quantity(10, is_buy, true, latest_obs);
    features.push(feature!(top_10_quantity_same_side));
    let top_25_quantity_same_side = top_n_quantity(25, is_buy, true, latest_obs);
    features.push(feature!(top_25_quantity_same_side));
    let top_1_quantity_opposite_side = top_n_quantity(1, is_buy, false, latest_obs);
    features.push(feature!(top_1_quantity_opposite_side));
    let top_5_quantity_opposite_side = top_n_quantity(5, is_buy, false, latest_obs);
    features.push(feature!(top_5_quantity_opposite_side));
    let top_10_quantity_opposite_side = top_n_quantity(10, is_buy, false, latest_obs);
    features.push(feature!(top_10_quantity_opposite_side));
    let top_25_quantity_opposite_side = top_n_quantity(25, is_buy, false, latest_obs);
    features.push(feature!(top_25_quantity_opposite_side));

    // Bid-ask imbalance
    let top_1_bid_ask_imbalance = top_n_bid_ask_imbalance(1, latest_obs);
    features.push(feature!(top_1_bid_ask_imbalance));
    let top_5_bid_ask_imbalance = top_n_bid_ask_imbalance(5, latest_obs);
    features.push(feature!(top_5_bid_ask_imbalance));
    let top_10_bid_ask_imbalance = top_n_bid_ask_imbalance(10, latest_obs);
    features.push(feature!(top_10_bid_ask_imbalance));
    let top_25_bid_ask_imbalance = top_n_bid_ask_imbalance(25, latest_obs);
    features.push(feature!(top_25_bid_ask_imbalance));

    // Bid and ask imbalance
    let bid_imbalance_5_to_5 = bid_or_ask_imbalance(true, 5, 10, latest_obs);
    features.push(feature!(bid_imbalance_5_to_5));
    let ask_imbalance_5_to_5 = bid_or_ask_imbalance(false, 5, 10, latest_obs);
    features.push(feature!(ask_imbalance_5_to_5));
    let bid_imbalance_10_to_15 = bid_or_ask_imbalance(true, 10, 25, latest_obs);
    features.push(feature!(bid_imbalance_10_to_15));
    let ask_imbalance_10_to_15 = bid_or_ask_imbalance(false, 10, 25, latest_obs);
    features.push(feature!(ask_imbalance_10_to_15));

    // Volume order imbalance
    let volume_order_imbalance_100_ms_1_level = volume_order_imbalance(100, 1, obs);
    features.push(feature!(volume_order_imbalance_100_ms_1_level));
    let volume_order_imbalance_100_ms_5_level = volume_order_imbalance(100, 5, obs);
    features.push(feature!(volume_order_imbalance_100_ms_5_level));
    let volume_order_imbalance_100_ms_10_level = volume_order_imbalance(100, 10, obs);
    features.push(feature!(volume_order_imbalance_100_ms_10_level));
    let volume_order_imbalance_100_ms_25_level = volume_order_imbalance(100, 25, obs);
    features.push(feature!(volume_order_imbalance_100_ms_25_level));
    let volume_order_imbalance_500_ms_1_level = volume_order_imbalance(500, 1, obs);
    features.push(feature!(volume_order_imbalance_500_ms_1_level));
    let volume_order_imbalance_500_ms_5_level = volume_order_imbalance(500, 5, obs);
    features.push(feature!(volume_order_imbalance_500_ms_5_level));
    let volume_order_imbalance_500_ms_10_level = volume_order_imbalance(500, 10, obs);
    features.push(feature!(volume_order_imbalance_500_ms_10_level));
    let volume_order_imbalance_500_ms_25_level = volume_order_imbalance(500, 25, obs);
    features.push(feature!(volume_order_imbalance_500_ms_25_level));
    let volume_order_imbalance_1000_ms_1_level = volume_order_imbalance(1000, 1, obs);
    features.push(feature!(volume_order_imbalance_1000_ms_1_level));
    let volume_order_imbalance_1000_ms_5_level = volume_order_imbalance(1000, 5, obs);
    features.push(feature!(volume_order_imbalance_1000_ms_5_level));
    let volume_order_imbalance_1000_ms_10_level = volume_order_imbalance(1000, 10, obs);
    features.push(feature!(volume_order_imbalance_1000_ms_10_level));
    let volume_order_imbalance_1000_ms_25_level = volume_order_imbalance(1000, 25, obs);
    features.push(feature!(volume_order_imbalance_1000_ms_25_level));

    // Trade flow imbalance
    let trade_flow_imbalance_10s = trade_flow_imbalance(10, rt, order.submit_timestamp);
    features.push(feature!(trade_flow_imbalance_10s));
    let trade_flow_imbalance_15s = trade_flow_imbalance(15, rt, order.submit_timestamp);
    features.push(feature!(trade_flow_imbalance_15s));
    let trade_flow_imbalance_30s = trade_flow_imbalance(30, rt, order.submit_timestamp);
    features.push(feature!(trade_flow_imbalance_30s));
    let trade_flow_imbalance_60s = trade_flow_imbalance(60, rt, order.submit_timestamp);
    features.push(feature!(trade_flow_imbalance_60s));

    // Corwin-Schultz Bid-ask Spread Estimator
    let css_100 = css(100, obs, order.submit_timestamp);
    features.push(feature!(css_100));
    let css_500 = css(500, obs, order.submit_timestamp);
    features.push(feature!(css_500));
    let css_1000 = css(1000, obs, order.submit_timestamp);
    features.push(feature!(css_1000));
    let css_5000 = css(5000, obs, order.submit_timestamp);
    features.push(feature!(css_5000));
    let css_10000 = css(10000, obs, order.submit_timestamp);
    features.push(feature!(css_10000));
    let css_30000 = css(30000, obs, order.submit_timestamp);
    features.push(feature!(css_30000));

    // Relative Strength Index
    let relative_strength_10_6 = relative_strength_index(10, 6, rt, order.submit_timestamp);
    features.push(feature!(relative_strength_10_6));
    let relative_strength_12_5 = relative_strength_index(12, 5, rt, order.submit_timestamp);
    features.push(feature!(relative_strength_12_5));
    let relative_strength_15_4 = relative_strength_index(15, 4, rt, order.submit_timestamp);
    features.push(feature!(relative_strength_15_4));
    let relative_strength_20_3 = relative_strength_index(20, 3, rt, order.submit_timestamp);
    features.push(feature!(relative_strength_20_3));
    let relative_strength_30_2 = relative_strength_index(30, 2, rt, order.submit_timestamp);
    features.push(feature!(relative_strength_30_2));
    let relative_strength_60_1 = relative_strength_index(60, 1, rt, order.submit_timestamp);
    features.push(feature!(relative_strength_60_1));

    // ? Aggregate Impact Index

    // Return
    let ln_return_100 = ln_return(100, obs);
    features.push(feature!(ln_return_100));
    let ln_return_500 = ln_return(500, obs);
    features.push(feature!(ln_return_500));
    let ln_return_1000 = ln_return(1000, obs);
    features.push(feature!(ln_return_1000));
    let ln_return_1500 = ln_return(1500, obs);
    features.push(feature!(ln_return_1500));
    let ln_return_3000 = ln_return(3000, obs);
    features.push(feature!(ln_return_3000));
    let ln_return_6000 = ln_return(6000, obs);
    features.push(feature!(ln_return_6000));

    // Midprice Variance
    let midprice_variance_5 = midprice_variance(5, obs, order.submit_timestamp);
    features.push(feature!(midprice_variance_5));
    let midprice_variance_10 = midprice_variance(10, obs, order.submit_timestamp);
    features.push(feature!(midprice_variance_10));
    let midprice_variance_15 = midprice_variance(15, obs, order.submit_timestamp);
    features.push(feature!(midprice_variance_15));
    let midprice_variance_30 = midprice_variance(30, obs, order.submit_timestamp);
    features.push(feature!(midprice_variance_30));
    let midprice_variance_60 = midprice_variance(60, obs, order.submit_timestamp);
    features.push(feature!(midprice_variance_60));

    // Signed Trade Size Variance
    let signed_trade_size_variance_5 = signed_trade_size_variance(5, rt, order.submit_timestamp);
    features.push(feature!(signed_trade_size_variance_5));
    let signed_trade_size_variance_10 = signed_trade_size_variance(10, rt, order.submit_timestamp);
    features.push(feature!(signed_trade_size_variance_10));
    let signed_trade_size_variance_15 = signed_trade_size_variance(15, rt, order.submit_timestamp);
    features.push(feature!(signed_trade_size_variance_15));
    let signed_trade_size_variance_30 = signed_trade_size_variance(30, rt, order.submit_timestamp);
    features.push(feature!(signed_trade_size_variance_30));
    let signed_trade_size_variance_60 = signed_trade_size_variance(60, rt, order.submit_timestamp);
    features.push(feature!(signed_trade_size_variance_60));

    features
}

fn diff_same_side_price(is_buy: bool, submit_ou: &OrderUpdate) -> f32 {
    if is_buy {
        (submit_ou.price - submit_ou.bid_price).as_f32()
    } else {
        (submit_ou.ask_price - submit_ou.price).as_f32()
    }
}

fn top_n_quantity(
    n: usize,
    is_buy: bool,
    same_side: bool,
    latest_obs: &(Vec<(Price, i32)>, Vec<(Price, i32)>, DateTime<Utc>),
) -> i32 {
    let mut qty = 0;
    for i in 0..n {
        let bid_i = latest_obs.0.get(i).unwrap();
        let ask_i = latest_obs.1.get(i).unwrap();
        if is_buy {
            qty += if same_side { bid_i.1 } else { ask_i.1 };
        } else {
            qty += if same_side { ask_i.1 } else { bid_i.1 };
        };
    }
    qty
}

fn top_n_bid_ask_imbalance(
    n: usize,
    latest_obs: &(Vec<(Price, i32)>, Vec<(Price, i32)>, DateTime<Utc>),
) -> f64 {
    let mut sum_bid_vol = 0.;
    let mut sum_ask_vol = 0.;
    for i in 0..n {
        let bid_i = latest_obs.0.get(i).unwrap();
        let ask_i = latest_obs.1.get(i).unwrap();
        sum_bid_vol += bid_i.1 as f64;
        sum_ask_vol += ask_i.1 as f64;
    }
    (sum_bid_vol - sum_ask_vol) / (sum_bid_vol + sum_ask_vol)
}

/// * `m` - top number of price levels
/// * `n` - total number of price levels
fn bid_or_ask_imbalance(
    is_bid: bool,
    m: usize,
    n: usize,
    latest_obs: &(Vec<(Price, i32)>, Vec<(Price, i32)>, DateTime<Utc>),
) -> f64 {
    let mut first_few_levels_of_vol = 0;
    let mut last_few_levels_of_vol = 0;
    let mut total_vol = 0;
    for i in 0..m {
        let vol = if is_bid {
            latest_obs.0.get(i).unwrap().1
        } else {
            latest_obs.1.get(i).unwrap().1
        };
        first_few_levels_of_vol += vol;
        total_vol += vol;
    }
    for i in m..n {
        let vol = if is_bid {
            latest_obs.0.get(i).unwrap().1
        } else {
            latest_obs.1.get(i).unwrap().1
        };
        last_few_levels_of_vol += vol;
        total_vol += vol;
    }
    (first_few_levels_of_vol - last_few_levels_of_vol) as f64 / total_vol as f64
}

/// `time_interval_ms`: no less than 100, which is the granularity of `obs`
fn volume_order_imbalance(
    time_step_ms: usize,
    price_levels: usize,
    obs: &VecDeque<(Vec<(Price, i32)>, Vec<(Price, i32)>, DateTime<Utc>)>,
) -> i32 {
    let latest_obs = obs.back().unwrap();
    let second_latest_obs = obs.get(obs.len() - 1 - (time_step_ms / 100)).unwrap(); // TODO: Fix overflow

    // To calculate cum_delta_v_bids
    let mut sum_of_bid_prices_at_time_t = 0.;
    let mut sum_of_bid_prices_at_time_t_minus_1 = 0.;
    let mut sum_of_bid_volume_at_time_t = 0;
    let mut sum_of_bid_volume_at_time_t_minus_1 = 0;

    // To calculate cum_delta_v_asks
    let mut sum_of_ask_prices_at_time_t = 0.;
    let mut sum_of_ask_prices_at_time_t_minus_1 = 0.;
    let mut sum_of_ask_volume_at_time_t = 0;
    let mut sum_of_ask_volume_at_time_t_minus_1 = 0;

    for i in 0..price_levels {
        let time_t_bid_price = latest_obs.0.get(i).unwrap().0.as_f64();
        let time_t_minus_1_bid_price = second_latest_obs.0.get(i).unwrap().0.as_f64();
        let time_t_bid_volume = latest_obs.0.get(i).unwrap().1;
        let time_t_minus_1_bid_volume = second_latest_obs.0.get(i).unwrap().1;
        let time_t_ask_price = latest_obs.1.get(i).unwrap().0.as_f64();
        let time_t_minus_1_ask_price = second_latest_obs.1.get(i).unwrap().0.as_f64();
        let time_t_ask_volume = latest_obs.1.get(i).unwrap().1;
        let time_t_minus_1_ask_volume = second_latest_obs.1.get(i).unwrap().1;

        sum_of_bid_prices_at_time_t += time_t_bid_price;
        sum_of_bid_prices_at_time_t_minus_1 += time_t_minus_1_bid_price;
        sum_of_bid_volume_at_time_t += time_t_bid_volume;
        sum_of_bid_volume_at_time_t_minus_1 += time_t_minus_1_bid_volume;
        sum_of_ask_prices_at_time_t += time_t_ask_price;
        sum_of_ask_prices_at_time_t_minus_1 += time_t_minus_1_ask_price;
        sum_of_ask_volume_at_time_t += time_t_ask_volume;
        sum_of_ask_volume_at_time_t_minus_1 += time_t_minus_1_ask_volume;
    }

    // Cumulative change in volume of bids/asks across top n price_levels from t-1 to t
    let cum_delta_v_bids = if sum_of_bid_prices_at_time_t < sum_of_bid_prices_at_time_t_minus_1 {
        0
    } else if sum_of_bid_prices_at_time_t == sum_of_bid_prices_at_time_t_minus_1 {
        sum_of_bid_volume_at_time_t - sum_of_bid_volume_at_time_t_minus_1
    } else if sum_of_bid_prices_at_time_t > sum_of_bid_prices_at_time_t_minus_1 {
        sum_of_bid_volume_at_time_t
    } else {
        unreachable!();
    };
    let cum_delta_v_asks = if sum_of_ask_prices_at_time_t < sum_of_ask_prices_at_time_t_minus_1 {
        sum_of_ask_volume_at_time_t
    } else if sum_of_ask_prices_at_time_t == sum_of_ask_prices_at_time_t_minus_1 {
        sum_of_ask_volume_at_time_t - sum_of_ask_volume_at_time_t_minus_1
    } else if sum_of_ask_prices_at_time_t > sum_of_ask_prices_at_time_t_minus_1 {
        0
    } else {
        unreachable!();
    };

    cum_delta_v_bids - cum_delta_v_asks
}

/// `window` in seconds
fn trade_flow_imbalance(window: i64, rt: &VecDeque<Trade>, submit_timestamp: DateTime<Utc>) -> f64 {
    let mut buy_vol = 0;
    let mut sell_vol = 0;
    let mut total_vol = 0;

    for trade in rt.iter().rev() {
        if trade.timestamp < submit_timestamp - Duration::seconds(window) {
            break;
        }
        if trade.size > 0 {
            buy_vol += trade.size;
            total_vol += trade.size;
        } else if trade.size < 0 {
            sell_vol -= trade.size;
            total_vol -= trade.size;
        }
    }

    (buy_vol - sell_vol) as f64 / total_vol as f64
}

/// `interval_ms` must be a multiplier of `constants::ORDER_BOOK_SNAPSHOT_INTERVAL`
fn css(
    interval_ms: i64,
    obs: &VecDeque<(Vec<(Price, i32)>, Vec<(Price, i32)>, DateTime<Utc>)>,
    submit_timestamp: DateTime<Utc>,
) -> f64 {
    let beta = css_bid_ask_size_average_adjustment(interval_ms, obs, 1, submit_timestamp);
    let gamma = css_bid_ask_size_average_adjustment(interval_ms, obs, 2, submit_timestamp);
    let alpha = ((f64::sqrt(2. * beta) - f64::sqrt(beta)) / (3. - 2. * f64::sqrt(2.)))
        - f64::sqrt(gamma / (3. - 2. * f64::sqrt(2.)));
    2. * (alpha.exp() - 1.) / (1. + alpha.exp())
}

/// - beta: `rolling_window_length` = 1
/// - gamma: `rolling_window_length` = 1
fn css_bid_ask_size_average_adjustment(
    interval_ms: i64,
    obs: &VecDeque<(Vec<(Price, i32)>, Vec<(Price, i32)>, DateTime<Utc>)>,
    rolling_window_length: i64,
    submit_timestamp: DateTime<Utc>,
) -> f64 {
    let h_0 = css_bid_ask_vol_mean(
        obs,
        submit_timestamp - Duration::milliseconds(rolling_window_length * interval_ms),
        submit_timestamp,
        true,
    );
    let l_0 = css_bid_ask_vol_mean(
        obs,
        submit_timestamp - Duration::milliseconds(rolling_window_length * interval_ms),
        submit_timestamp,
        false,
    );
    let h_1 = css_bid_ask_vol_mean(
        obs,
        submit_timestamp - Duration::milliseconds((rolling_window_length + 1) * interval_ms),
        submit_timestamp - Duration::milliseconds(interval_ms),
        true,
    );
    let l_1 = css_bid_ask_vol_mean(
        obs,
        submit_timestamp - Duration::milliseconds((rolling_window_length + 1) * interval_ms),
        submit_timestamp - Duration::milliseconds(interval_ms),
        false,
    );

    ((h_0 / l_0).ln().powi(2) + (h_1 / l_1).ln().powi(2)) / 2.
}

/// Observed high/low of the mean of level-1 bid and ask size within time interval T, respectively.
fn css_bid_ask_vol_mean(
    obs: &VecDeque<(Vec<(Price, i32)>, Vec<(Price, i32)>, DateTime<Utc>)>,
    from_ts: DateTime<Utc>,
    to_ts: DateTime<Utc>,
    is_high: bool,
) -> f64 {
    let mut observed_high_or_low_bid_ask_vol_mean = if is_high { f64::MIN } else { f64::MAX }; // top-level volume only

    for snapshot in obs.iter().rev() {
        if snapshot.2 < from_ts || snapshot.2 > to_ts {
            continue;
        }
        let bid_ask_vol_mean =
            (snapshot.0.first().unwrap().1 + snapshot.1.first().unwrap().1) as f64 / 2.;
        observed_high_or_low_bid_ask_vol_mean = if is_high {
            f64::max(observed_high_or_low_bid_ask_vol_mean, bid_ask_vol_mean)
        } else {
            f64::min(observed_high_or_low_bid_ask_vol_mean, bid_ask_vol_mean)
        };
    }

    observed_high_or_low_bid_ask_vol_mean
}

/// For each period, compare this period VWAP to last period VWAP, if increment, avg_up ++, else avg_down ++
///
/// `periods` * `time_interval_ms` should not exceed 60 seconds
fn relative_strength_index(
    periods: i64,
    time_interval_s: i64,
    rt: &VecDeque<Trade>,
    submit_timestamp: DateTime<Utc>,
) -> f64 {
    // Put trades into bins of length time_interval_s
    let mut trade_bins = Vec::new();
    for i in 0..periods {
        let start_ts = submit_timestamp - Duration::seconds(time_interval_s * (i + 1));
        let end_ts = submit_timestamp - Duration::seconds(time_interval_s * i);

        let mut trade_bin_i = Vec::new();
        for trade in rt {
            if trade.timestamp > start_ts && trade.timestamp <= end_ts {
                trade_bin_i.push(trade);
            }
        }

        trade_bins.push(trade_bin_i);
    }
    trade_bins.reverse(); // Latest trades at the end of the Vec

    // Calculate VWAP of ask price for each bin
    let mut vwaps = Vec::new();
    for trade_bin in trade_bins.clone() {
        let vwap = if trade_bin.is_empty() {
            None
        } else {
            let mut sum_price_times_vol = 0.;
            let mut total_vol = 0.;

            for trade in trade_bin {
                if trade.size > 0 {
                    let vol = trade.size as f64;
                    sum_price_times_vol += trade.price.as_f64() * vol;
                    total_vol += vol;
                }
            }

            Some(sum_price_times_vol / total_vol)
        };

        vwaps.push(vwap);
    }

    // Calc avg gain/up and avg loss/down
    let mut up_count = 0.;
    let mut down_count = 0.;
    let total_count = (vwaps.len() - 1) as f64;

    let mut opt_last_vwap = None;
    for opt_vwap in vwaps.clone() {
        match opt_vwap {
            Some(vwap) if !vwap.is_nan() => {
                if let Some(last_vwap) = opt_last_vwap {
                    if vwap > last_vwap {
                        up_count += 1.;
                    } else if vwap < last_vwap {
                        down_count += 1.;
                    }
                }

                opt_last_vwap = opt_vwap;
            }
            _ => {}
        }
    }

    let avg_up = up_count / total_count;
    let avg_down = down_count / total_count;

    100. - (100. / (1. + avg_up / avg_down))
}

fn ln_return(
    window_ms: i64,
    obs: &VecDeque<(Vec<(Price, i32)>, Vec<(Price, i32)>, DateTime<Utc>)>,
) -> f64 {
    let interval_multiplier = window_ms / ORDER_BOOK_SNAPSHOT_INTERVAL;
    let last_snapshot = obs.back().unwrap();
    let second_last_snapshot = obs
        .get(obs.len() - 1 - interval_multiplier as usize)
        .unwrap();

    let midprice_t = (last_snapshot.0.first().unwrap().0.as_f64()
        + last_snapshot.1.first().unwrap().0.as_f64())
        / 2.;
    let midprice_t_minus_w_plus_1 = (second_last_snapshot.0.first().unwrap().0.as_f64()
        + second_last_snapshot.1.first().unwrap().0.as_f64())
        / 2.;

    midprice_t.ln() - midprice_t_minus_w_plus_1.ln()
}

fn midprice_variance(
    window_s: i64,
    obs: &VecDeque<(Vec<(Price, i32)>, Vec<(Price, i32)>, DateTime<Utc>)>,
    submit_timestamp: DateTime<Utc>,
) -> f64 {
    let mut midprice_vals = Vec::new();
    for snapshot in obs.iter().rev() {
        if snapshot.2 < submit_timestamp - Duration::seconds(window_s) {
            break;
        }
        let midprice =
            (snapshot.0.first().unwrap().0.as_f64() + snapshot.1.first().unwrap().0.as_f64()) / 2.;
        midprice_vals.push(midprice);
    }

    variance(midprice_vals)
}

fn signed_trade_size_variance(
    window_s: i64,
    rt: &VecDeque<Trade>,
    submit_timestamp: DateTime<Utc>,
) -> f64 {
    let mut size_vals = Vec::new();
    for trade in rt.iter().rev() {
        if trade.timestamp < submit_timestamp - Duration::seconds(window_s) {
            break;
        }
        size_vals.push(trade.size as f64);
    }

    variance(size_vals)
}

fn variance(vals: Vec<f64>) -> f64 {
    let mean = vals.iter().sum::<f64>() / vals.len() as f64;
    vals.iter().map(|val| (mean - val).powi(2)).sum::<f64>() / vals.len() as f64
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        constants::DB_URL,
        Exchange::{self, *},
    };
    use lazy_static::lazy_static;
    use sqlx::{
        postgres::{PgPoolOptions, PgRow},
        Row,
    };
    use std::collections::HashMap;
    use tokio::runtime::Runtime;

    lazy_static! {
        static ref ROWS: Vec<OrdersMlTestRow> = {
            let mut rt = Runtime::new().unwrap();
            rt.block_on(get_rows())
        };
    }

    #[derive(Debug, Clone)]
    struct OrdersMlTestRow {
        exchange: Exchange,
        quantity: i32,
        order: Order,
        obs: VecDeque<(Vec<(Price, i32)>, Vec<(Price, i32)>, DateTime<Utc>)>,
        rt: VecDeque<Trade>,
        features: HashMap<String, Value>,
        submit_timestamp: DateTime<Utc>,
    }

    async fn get_rows() -> Vec<OrdersMlTestRow> {
        let pool = PgPoolOptions::new()
            .max_connections(20)
            .connect(DB_URL)
            .await
            .unwrap();
        sqlx::query("SELECT * FROM orders_ml_test")
            .map(|row: PgRow| {
                let exchange_string: String = row.get(1);
                let exchange = match exchange_string.as_str() {
                    "Bitmex" => Bitmex,
                    "Deribit" => Deribit,
                    err => panic!("{:?}", err),
                };
                let quantity: i32 = row.get(2);
                let order_json: Value = row.get(3);
                let order: Order = serde_json::from_value(order_json).unwrap();
                let obs_json: Value = row.get(4);
                let obs: VecDeque<(Vec<(Price, i32)>, Vec<(Price, i32)>, DateTime<Utc>)> =
                    serde_json::from_value(obs_json).unwrap();
                let rt_json: Value = row.get(5);
                let rt: VecDeque<Trade> = serde_json::from_value(rt_json).unwrap();
                let features: HashMap<String, Value> = {
                    let mut features = HashMap::new();
                    let features_json: Value = row.get(6);

                    for feature in features_json.as_array().unwrap() {
                        let feature_name = feature["name"].as_str().unwrap().to_string();
                        let feature_val = feature["value"].clone();
                        features.insert(feature_name, feature_val);
                    }

                    features
                };
                let submit_timestamp: DateTime<Utc> = row.get(7);
                OrdersMlTestRow {
                    exchange,
                    quantity,
                    order,
                    obs,
                    rt,
                    features,
                    submit_timestamp,
                }
            })
            .fetch_all(&pool)
            .await
            .unwrap()
    }

    #[test]
    fn test_bid_ask_spread() {
        for row in ROWS.iter() {
            let calc_val = {
                let order = &row.order;
                let submit_ou = order.updates.get(0).unwrap();
                (submit_ou.ask_price - submit_ou.bid_price).as_f64()
            };
            let db_val = get_feature_as_f64(&row.features, "bid_ask_spread".to_string());
            assert_eq!(calc_val, db_val);
        }
    }

    #[test]
    fn test_diff_same_side_price() {
        for row in ROWS.iter() {
            let calc_val = {
                let order = &row.order;
                let submit_ou = order.updates.get(0).unwrap();
                if order.quantity > 0 {
                    (order.submit_price - submit_ou.bid_price).as_f64()
                } else {
                    (submit_ou.ask_price - order.submit_price).as_f64()
                }
            };
            let db_val = get_feature_as_f64(&row.features, "diff_same_side_price".to_string());

            assert_eq!(calc_val, db_val);
        }
    }

    #[test]
    fn test_top_n_quantity() {
        for row in ROWS.iter() {
            let ns = vec![1, 5, 10, 25];
            let is_buy = row.order.quantity > 0;
            let latest_obs = row.obs.back().unwrap();

            for n in ns {
                let mut top_n_quantity_same_side_calc = 0;
                let mut top_n_quantity_opposite_side_calc = 0;

                for i in 0..n {
                    if is_buy {
                        top_n_quantity_same_side_calc += latest_obs.0.get(i).unwrap().1;
                        top_n_quantity_opposite_side_calc += latest_obs.1.get(i).unwrap().1;
                    } else {
                        top_n_quantity_same_side_calc += latest_obs.1.get(i).unwrap().1;
                        top_n_quantity_opposite_side_calc += latest_obs.0.get(i).unwrap().1;
                    };
                }

                let top_n_quantity_same_side_db =
                    get_feature_as_i64(&row.features, format!("top_{}_quantity_same_side", n))
                        as i32;
                let top_n_quantity_opposite_side_db =
                    get_feature_as_i64(&row.features, format!("top_{}_quantity_opposite_side", n))
                        as i32;

                assert_eq!(top_n_quantity_same_side_calc, top_n_quantity_same_side_db);
                assert_eq!(
                    top_n_quantity_opposite_side_calc,
                    top_n_quantity_opposite_side_db
                );
            }
        }
    }

    #[test]
    fn test_top_n_bid_ask_imbalance() {
        let ns = [1, 5, 10, 25];

        for row in ROWS.iter() {
            let latest_snapshot = row.obs.back().unwrap();
            let bids = &latest_snapshot.0;
            let asks = &latest_snapshot.1;

            for n in &ns {
                let mut top_n_bids = 0.;
                let mut top_n_asks = 0.;

                for i in 0..*n {
                    top_n_bids += bids.get(i).unwrap().1 as f64;
                    top_n_asks += asks.get(i).unwrap().1 as f64;
                }

                let top_n_bid_ask_imbalance_calc =
                    (top_n_bids - top_n_asks) / (top_n_bids + top_n_asks);
                let top_n_bid_ask_imbalance_db =
                    get_feature_as_f64(&row.features, format!("top_{}_bid_ask_imbalance", n));

                assert!((top_n_bid_ask_imbalance_calc - top_n_bid_ask_imbalance_db).abs() < 1e-15);
            }
        }
    }

    #[test]
    fn test_bid_or_ask_imbalance() {
        let ratios = [(5, 5), (10, 15)];

        for row in ROWS.iter() {
            let last_obs = row.obs.back().unwrap();
            let bids = &last_obs.0;
            let asks = &last_obs.1;

            for ratio in ratios.iter() {
                let m = ratio.0; // m = top levels
                let n = ratio.1; // n = bottom levels

                let mut sum_m_bids = 0.;
                let mut sum_n_bids = 0.;

                let mut sum_m_asks = 0.;
                let mut sum_n_asks = 0.;

                for i in 0..(m + n) {
                    if i < m {
                        sum_m_bids += bids.get(i).unwrap().1 as f64;
                        sum_m_asks += asks.get(i).unwrap().1 as f64;
                    } else if i < m + n {
                        sum_n_bids += bids.get(i).unwrap().1 as f64;
                        sum_n_asks += asks.get(i).unwrap().1 as f64;
                    }
                }

                let bid_imbalance_m_to_n_calc =
                    (sum_m_bids - sum_n_bids) / (sum_m_bids + sum_n_bids);
                let ask_imbalance_m_to_n_calc =
                    (sum_m_asks - sum_n_asks) / (sum_m_asks + sum_n_asks);

                let bid_imbalance_m_to_n_db =
                    get_feature_as_f64(&row.features, format!("bid_imbalance_{}_to_{}", m, n));
                let ask_imbalance_m_to_n_db =
                    get_feature_as_f64(&row.features, format!("ask_imbalance_{}_to_{}", m, n));

                assert!((bid_imbalance_m_to_n_calc - bid_imbalance_m_to_n_db).abs() < 1e-15);
                assert!((ask_imbalance_m_to_n_calc - ask_imbalance_m_to_n_db).abs() < 1e-15);
            }
        }
    }

    #[test]
    fn test_volume_order_imbalance() {
        let ws = [100, 500, 1000]; // in ms
        let ns = [1, 5, 10, 25];

        for row in ROWS.iter() {
            let t_snapshot = row.obs.back().unwrap();

            for w in ws.iter() {
                let t_minus_w_snapshot = row
                    .obs
                    .get(row.obs.len() - 1 - (*w as usize / 100))
                    .unwrap();

                for n in ns.iter() {
                    let mut sum_t_bid_price = 0.;
                    let mut sum_t_bid_vol = 0;
                    let mut sum_t_ask_price = 0.;
                    let mut sum_t_ask_vol = 0;
                    let mut sum_t_minus_w_bid_price = 0.;
                    let mut sum_t_minus_w_bid_vol = 0;
                    let mut sum_t_minus_w_ask_price = 0.;
                    let mut sum_t_minus_w_ask_vol = 0;

                    for i in 0..*n {
                        let t_bid_price = t_snapshot.0.get(i).unwrap().0.as_f64();
                        let t_bid_vol = t_snapshot.0.get(i).unwrap().1;
                        let t_ask_price = t_snapshot.1.get(i).unwrap().0.as_f64();
                        let t_ask_vol = t_snapshot.1.get(i).unwrap().1;
                        let t_minus_w_bid_price = t_minus_w_snapshot.0.get(i).unwrap().0.as_f64();
                        let t_minus_w_bid_vol = t_minus_w_snapshot.0.get(i).unwrap().1;
                        let t_minus_w_ask_price = t_minus_w_snapshot.1.get(i).unwrap().0.as_f64();
                        let t_minus_w_ask_vol = t_minus_w_snapshot.1.get(i).unwrap().1;

                        sum_t_bid_price += t_bid_price;
                        sum_t_bid_vol += t_bid_vol;
                        sum_t_ask_price += t_ask_price;
                        sum_t_ask_vol += t_ask_vol;
                        sum_t_minus_w_bid_price += t_minus_w_bid_price;
                        sum_t_minus_w_bid_vol += t_minus_w_bid_vol;
                        sum_t_minus_w_ask_price += t_minus_w_ask_price;
                        sum_t_minus_w_ask_vol += t_minus_w_ask_vol;
                    }

                    let delta_bid_vol = if sum_t_bid_price < sum_t_minus_w_bid_price {
                        0
                    } else if sum_t_bid_price == sum_t_minus_w_bid_price {
                        sum_t_bid_vol - sum_t_minus_w_bid_vol
                    } else {
                        sum_t_bid_vol
                    };
                    let delta_ask_vol = if sum_t_ask_price < sum_t_minus_w_ask_price {
                        sum_t_ask_vol
                    } else if sum_t_ask_price == sum_t_minus_w_ask_price {
                        sum_t_ask_vol - sum_t_minus_w_ask_vol
                    } else {
                        0
                    };

                    let voi_w_n_calc = (delta_bid_vol - delta_ask_vol) as i64;
                    let voi_w_n_db = get_feature_as_i64(
                        &row.features,
                        format!("volume_order_imbalance_{}_ms_{}_level", w, n),
                    );

                    assert_eq!(voi_w_n_calc, voi_w_n_db);
                }
            }
        }
    }

    #[test]
    fn test_trade_flow_imbalance() {
        let ws = [10, 15, 30, 60];

        for row in ROWS.iter() {
            for w in &ws {
                let mut sum_buy_vol = 0;
                let mut sum_sell_vol = 0;

                for trade in &row.rt {
                    if trade.timestamp < row.order.submit_timestamp - Duration::seconds(*w) {
                        continue;
                    }

                    if trade.size > 0 {
                        sum_buy_vol += trade.size;
                    } else {
                        sum_sell_vol -= trade.size;
                    }
                }

                let tfi_calc =
                    (sum_buy_vol - sum_sell_vol) as f64 / (sum_buy_vol + sum_sell_vol) as f64;
                let tfi_db =
                    get_feature_as_f64(&row.features, format!("trade_flow_imbalance_{}s", w));

                assert!((tfi_calc - tfi_db).abs() < 1e-15);
            }
        }
    }

    #[test]
    fn test_css() {
        let interval_ms_vals = [100, 500, 1000, 5000, 10000, 30000];

        for row in ROWS.iter() {
            for interval_ms in &interval_ms_vals {
                let interval_multiplier = (interval_ms / ORDER_BOOK_SNAPSHOT_INTERVAL) as usize;

                let (low_0_1, high_0_1) = css_low_high(interval_multiplier, 0, 1, &row.obs);
                let (low_1_2, high_1_2) = css_low_high(interval_multiplier, 1, 2, &row.obs);
                let (low_0_2, high_0_2) = css_low_high(interval_multiplier, 0, 2, &row.obs);
                let (low_1_3, high_1_3) = css_low_high(interval_multiplier, 1, 3, &row.obs);

                let beta =
                    ((high_0_1 / low_0_1).ln().powi(2) + (high_1_2 / low_1_2).ln().powi(2)) / 2.;
                let gamma =
                    ((high_0_2 / low_0_2).ln().powi(2) + (high_1_3 / low_1_3).ln().powi(2)) / 2.;
                let alpha = (f64::sqrt(2. * beta) - f64::sqrt(beta)) / (3. - 2. * f64::sqrt(2.))
                    - f64::sqrt(gamma / (3. - 2. * f64::sqrt(2.)));

                let css_calc = 2. * (alpha.exp() - 1.) / (1. + alpha.exp());
                let css_db = get_feature_as_f64(&row.features, format!("css_{}", interval_ms));

                println!("css_calc = {:?}", css_calc);
                println!("css_db = {:?}", css_db);

                assert!((css_calc - css_db).abs() < 1e-15);
            }
        }
    }

    fn css_low_high(
        interval_multiplier: usize,
        from_index: usize,
        to_index: usize,
        obs: &VecDeque<(Vec<(Price, i32)>, Vec<(Price, i32)>, DateTime<Utc>)>,
    ) -> (f64, f64) {
        let mut low = f64::MAX;
        let mut high = f64::MIN;

        for (i, snapshot) in obs.iter().rev().enumerate() {
            if i < from_index * interval_multiplier {
                continue;
            } else if i >= to_index * interval_multiplier {
                break;
            }

            let mean_bid_ask_vol =
                (snapshot.0.first().unwrap().1 + snapshot.1.first().unwrap().1) as f64 / 2.;
            low = f64::min(low, mean_bid_ask_vol);
            high = f64::max(high, mean_bid_ask_vol);
        }

        (low, high)
    }

    #[test]
    fn test_relative_strength_index() {
        let periods_and_lengths = [(10, 6), (12, 5), (15, 4), (20, 3), (30, 2), (60, 1)];

        for row in ROWS.iter() {
            for (periods, period_length) in periods_and_lengths.iter() {
                let mut trade_bins = Vec::new();
                for _ in 0..*periods {
                    trade_bins.push(Vec::new());
                }

                for trade in &row.rt {
                    for i in 0..*periods {
                        let start_ts =
                            row.order.submit_timestamp - Duration::seconds((i + 1) * period_length);
                        let end_ts =
                            row.order.submit_timestamp - Duration::seconds(i * period_length);
                        if trade.timestamp > start_ts && trade.timestamp <= end_ts {
                            trade_bins.get_mut(i as usize).unwrap().push(trade);
                        }
                    }
                }
                trade_bins.reverse();

                let mut opt_vwaps = Vec::new();
                for trade_bin in trade_bins {
                    if trade_bin.is_empty() {
                        opt_vwaps.push(None);
                        continue;
                    }

                    let mut sum_vol_times_p = 0.;
                    let mut sum_vol = 0.;

                    for trade in trade_bin {
                        if trade.size > 0 {
                            sum_vol_times_p += trade.size as f64 * trade.price.as_f64();
                            sum_vol += trade.size as f64;
                        }
                    }

                    opt_vwaps.push(Some(sum_vol_times_p / sum_vol));
                }

                let mut gain_count = 0.;
                let mut loss_count = 0.;

                let mut opt_last_vwap = None;
                for opt_vwap in &opt_vwaps {
                    match opt_vwap {
                        Some(vwap) if !vwap.is_nan() => {
                            if let Some(last_vwap) = opt_last_vwap {
                                if vwap > last_vwap {
                                    gain_count += 1.;
                                } else if vwap < last_vwap {
                                    loss_count += 1.;
                                }
                            }

                            opt_last_vwap = Some(vwap);
                        }
                        _ => {}
                    }
                }

                let total_count = opt_vwaps.len() - 1;
                let avg_gain = gain_count / total_count as f64;
                let avg_loss = loss_count / total_count as f64;

                let rsi_calc = 100. - 100. / (1. + (avg_gain / avg_loss));
                let rsi_db = get_feature_as_f64(
                    &row.features,
                    format!("relative_strength_{}_{}", periods, period_length),
                );

                assert!((rsi_calc - rsi_db).abs() < 1e-14);
            }
        }
    }

    #[test]
    fn test_ln_return() {
        let time_diffs = vec![100, 500, 1000, 1500, 3000, 6000];

        for row in ROWS.iter() {
            for time_diff in &time_diffs {
                let last_snapshot = row.obs.back().unwrap();
                let second_last_snapshot = row
                    .obs
                    .get(row.obs.len() - 1 - (time_diff / ORDER_BOOK_SNAPSHOT_INTERVAL) as usize)
                    .unwrap();

                let ln_return_t_calc = ((last_snapshot.0.first().unwrap().0
                    + last_snapshot.1.get(0).unwrap().0)
                    .as_f64()
                    / 2.)
                    .ln();
                let ln_return_t_minus_calc = ((second_last_snapshot.0.get(0).unwrap().0
                    + second_last_snapshot.1.get(0).unwrap().0)
                    .as_f64()
                    / 2.)
                    .ln();
                let ln_return_calc = ln_return_t_calc - ln_return_t_minus_calc;
                let ln_return_db =
                    get_feature_as_f64(&row.features, format!("ln_return_{}", time_diff));

                assert!((ln_return_calc - ln_return_db).abs() < 1e-19); // approx equal for floats
            }
        }
    }

    #[test]
    fn test_midprice_variance() {
        let window_lengths = [5, 10, 15, 30, 60]; // in seconds

        for row in ROWS.iter() {
            for window_length in window_lengths.iter() {
                let mut midprices: Vec<f64> = Vec::new();

                for snapshot in &row.obs {
                    if snapshot.2 < row.submit_timestamp - Duration::seconds(*window_length) {
                        continue;
                    }
                    let midprice = (snapshot.0.first().unwrap().0.as_f64()
                        + snapshot.1.first().unwrap().0.as_f64())
                        / 2.;
                    midprices.push(midprice);
                }

                let calc_val = variance(midprices);
                let db_val: f64 = get_feature_as_f64(
                    &row.features,
                    format!("midprice_variance_{}", window_length),
                );

                assert!((calc_val - db_val).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn test_signed_trade_size_variance() {
        let window_lengths = [5, 10, 15, 30, 60]; // in seconds

        for row in ROWS.iter() {
            for window_length in window_lengths.iter() {
                let mut trade_sizes: Vec<f64> = Vec::new();

                for trade in &row.rt {
                    if trade.timestamp < row.submit_timestamp - Duration::seconds(*window_length) {
                        continue;
                    }
                    trade_sizes.push(trade.size as f64);
                }

                let calc_val = variance(trade_sizes);
                let db_val: f64 = get_feature_as_f64(
                    &row.features,
                    format!("signed_trade_size_variance_{}", window_length),
                );

                assert!(calc_val.is_nan() && db_val.is_nan() || (calc_val - db_val).abs() < 1e-5);
            }
        }
    }
}

pub fn get_feature_as_i64(features: &HashMap<String, Value>, key: String) -> i64 {
    features.get(key.as_str()).unwrap().as_i64().unwrap()
}

pub fn get_feature_as_f64(features: &HashMap<String, Value>, key: String) -> f64 {
    features
        .get(key.as_str())
        .unwrap()
        .as_f64()
        .unwrap_or(f64::NAN)
}
