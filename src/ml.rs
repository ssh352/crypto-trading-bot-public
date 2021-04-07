use crate::Exchange::{self, *};

// const CHOSEN_FEATURES: &[&str] = &[
//     "bid_ask_spread",
//     "top_1_quantity_same_side",
//     "top_1_quantity_opposite_side",
//     "top_1_bid_ask_imbalance",
//     "bid_imbalance_10_to_15",
//     "ask_imbalance_10_to_15",
//     "volume_order_imbalance_1000_ms_1_level",
//     "trade_flow_imbalance_60s",
//     "css_1000",
//     "relative_strength_60_1",
//     "ln_return_6000",
//     "midprice_variance_60",
//     "signed_trade_size_variance_60",
// ];

pub fn fill_prob(exchange: Exchange, is_buy: bool, features: Vec<f64>) -> f64 {
    let (intercept, coefficients) = match exchange {
        Bitmex if is_buy => (
            5.6031972e-09,
            vec![
                4.19813533e-09,
                -5.90009761e-07,
                4.06304614e-07,
                -3.85165034e-09,
                -3.04360519e-09,
                2.07887165e-10,
                -3.56927140e-07,
                5.30108840e-10,
                -8.34661357e-09,
                2.97566797e-07,
                3.55728616e-13,
                1.32264941e-05,
                8.75037676e-12,
            ],
        ),
        Bitmex if !is_buy => (
            1.41198256e-07,
            vec![
                8.53349139e-08,
                -6.87256344e-07,
                4.63254671e-07,
                -2.39471475e-09,
                -5.48042602e-08,
                -2.97420022e-08,
                -3.68332346e-08,
                -1.43544515e-08,
                -2.03481892e-07,
                7.57006014e-06,
                -3.55067667e-12,
                1.87815739e-04,
                2.38165283e-12,
            ],
        ),
        Deribit if is_buy => (
            2.5342715e-12,
            vec![
                7.78016977e-12,
                -8.00969944e-07,
                3.60815148e-07,
                -4.99978971e-13,
                -1.36792488e-12,
                -1.10975075e-12,
                -3.28471664e-08,
                -7.91816002e-14,
                -4.01531699e-12,
                1.56055589e-10,
                5.12505737e-17,
                4.08535250e-09,
                1.06221974e-11,
            ],
        ),
        Deribit if !is_buy => (
            3.40547606e-12,
            vec![
                1.27911598e-11,
                -9.29507940e-07,
                2.88116720e-07,
                -7.74169447e-14,
                -1.86438454e-12,
                -1.73525247e-12,
                -1.65775906e-08,
                -1.27147865e-13,
                -5.47842719e-12,
                2.13725169e-10,
                -4.11693760e-17,
                4.76134317e-09,
                1.25278597e-11,
            ],
        ),
        err => panic!("{:?}", err),
    };
    let x = intercept
        + coefficients
            .iter()
            .zip(features)
            .map(|(a, b)| a * b)
            .sum::<f64>();
    sigmoid(x)
}

fn sigmoid(x: f64) -> f64 {
    1. / (1. + (-x).exp())
}
