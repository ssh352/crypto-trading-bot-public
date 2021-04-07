use crate::{
    constants::*,
    order::{Order, OrderStatus, OrderUpdate},
    Exchange::*,
};
use chrono::{DateTime, Utc};
use std::cmp::{max, min};

/// Keeps track of current state of portfolio
#[derive(Clone, Debug)]
pub struct Portfolio {
    pub bitmex_orders: Vec<Order>,
    pub deribit_orders: Vec<Order>,
}

impl Portfolio {
    pub fn new() -> Portfolio {
        Portfolio {
            bitmex_orders: vec![],
            deribit_orders: vec![],
        }
    }

    pub fn on_order_update(&mut self, ou: OrderUpdate) {
        let ou_clone = ou.clone();

        let orders = match ou.exchange {
            Bitmex => &mut self.bitmex_orders,
            Deribit => &mut self.deribit_orders,
        };

        match ou.status {
            OrderStatus::Submitted => {
                orders.push(Order {
                    exchange: ou.exchange,
                    id: ou.id.clone(),
                    quantity: ou.quantity,
                    submit_price: ou.price,
                    submit_timestamp: ou.timestamp,
                    updates: vec![ou],
                });
            }
            _ => {
                orders.last_mut().unwrap().updates.push(ou);
            }
        }

        let latest_delta_str = match self.latest_delta() {
            Some(latest_delta) => latest_delta.to_string(),
            None => "".to_string(),
        };

        println!(
            "\r[{}] {:3} * {:7} @ {:7} {:9} {:.9}                                                                                                ",
            ou_clone.timestamp.format("%Y-%m-%d %H:%M:%S%.3f"),
            ou_clone.quantity,
            ou_clone.exchange.to_string().as_str(),
            ou_clone.price.to_string(),
            format!("{:?}", ou_clone.status),
            latest_delta_str
        );
        if self.is_filled()
            && self.current_bitmex_filled_pos() == 0
            && self.current_deribit_filled_pos() == 0
        {
            println!("----------------------------------------------------------------");
            println!(
                "[{}] Satoshi: {:.2}, USD/Year: {:.2}",
                max(
                    self.bitmex_orders
                        .last()
                        .unwrap()
                        .updates
                        .last()
                        .unwrap()
                        .timestamp,
                    self.deribit_orders
                        .last()
                        .unwrap()
                        .updates
                        .last()
                        .unwrap()
                        .timestamp
                )
                .format("%Y-%m-%d %H:%M:%S%.3f"),
                self.profit_in_btc() / 0.00000001,
                self.profit_per_year_in_usd(ou_clone.timestamp)
            );
            println!("================================================================");
            // std::process::exit(0);
        }
    }

    pub fn profit_in_btc(&self) -> f64 {
        let mut profit = 0.;
        for (bitmex_order, deribit_order) in self.bitmex_orders.iter().zip(&self.deribit_orders) {
            if let (Some(bou), Some(dou)) =
                (bitmex_order.updates.last(), deribit_order.updates.last())
            {
                if bou.status == OrderStatus::Filled && dou.status == OrderStatus::Filled {
                    profit += bou.quantity as f64 / bou.price.as_f64();
                    profit += dou.quantity as f64 / dou.price.as_f64();
                    profit += bou.quantity.abs() as f64 / bou.price.as_f64() * BITMEX_REBATE_RATE;
                }
            }
        }
        profit
    }

    pub fn profit_per_year_in_usd(&self, ts: DateTime<Utc>) -> f64 {
        if self.bitmex_orders.is_empty() || self.deribit_orders.is_empty() {
            return f64::NAN;
        }

        let first_ts = min(
            self.bitmex_orders.first().unwrap().submit_timestamp,
            self.deribit_orders.first().unwrap().submit_timestamp,
        );

        self.profit_in_btc() * 267. * 35000. * 31556952000.
            / (ts - first_ts).num_milliseconds() as f64
    }

    pub fn position(&self) -> Option<i32> {
        if self.bitmex_orders.len() == self.deribit_orders.len() {
            if let Some(last_bitmex_order) = self.bitmex_orders.last() {
                return Some(last_bitmex_order.quantity);
            }
        } else {
            unreachable!();
        }

        None
    }

    pub fn is_empty(&self) -> bool {
        self.bitmex_orders.is_empty() && self.deribit_orders.is_empty()
    }

    pub fn is_one_filled_one_unfilled(&self) -> bool {
        if self.bitmex_orders.len() == self.deribit_orders.len() {
            if let (Some(last_bitmex_order), Some(last_deribit_order)) =
                (self.bitmex_orders.last(), self.deribit_orders.last())
            {
                if let (Some(last_bitmex_ou), Some(last_deribit_ou)) = (
                    last_bitmex_order.updates.last(),
                    last_deribit_order.updates.last(),
                ) {
                    return last_bitmex_ou.status == OrderStatus::Filled
                        && last_deribit_ou.status != OrderStatus::Filled
                        || last_bitmex_ou.status != OrderStatus::Filled
                            && last_deribit_ou.status == OrderStatus::Filled;
                }
            }
        }

        false
    }

    pub fn is_filled(&self) -> bool {
        if self.bitmex_orders.len() == self.deribit_orders.len() {
            if let (Some(last_bitmex_order), Some(last_deribit_order)) =
                (self.bitmex_orders.last(), self.deribit_orders.last())
            {
                if let (Some(last_bitmex_ou), Some(last_deribit_ou)) = (
                    last_bitmex_order.updates.last(),
                    last_deribit_order.updates.last(),
                ) {
                    return last_bitmex_ou.status == OrderStatus::Filled
                        && last_deribit_ou.status == OrderStatus::Filled;
                }
            }
        }

        false
    }

    pub fn is_long(&self) -> bool {
        if self.bitmex_orders.len() == self.deribit_orders.len() {
            if let (Some(last_bitmex_order), Some(last_deribit_order)) =
                (self.bitmex_orders.last(), self.deribit_orders.last())
            {
                return last_bitmex_order.quantity > 0 && last_deribit_order.quantity < 0;
            }
        }

        false
    }

    pub fn is_short(&self) -> bool {
        if self.bitmex_orders.len() == self.deribit_orders.len() {
            if let (Some(last_bitmex_order), Some(last_deribit_order)) =
                (self.bitmex_orders.last(), self.deribit_orders.last())
            {
                return last_bitmex_order.quantity < 0 && last_deribit_order.quantity > 0;
            }
        }

        false
    }

    pub fn current_bitmex_filled_pos(&self) -> i32 {
        let mut pos = 0;
        for order in &self.bitmex_orders {
            let last_ou = order.updates.last().unwrap();
            if last_ou.status == OrderStatus::Filled {
                pos += last_ou.quantity;
            }
        }
        pos
    }

    pub fn current_deribit_filled_pos(&self) -> i32 {
        let mut pos = 0;
        for order in &self.deribit_orders {
            let last_ou = order.updates.last().unwrap();
            if last_ou.status == OrderStatus::Filled {
                pos += last_ou.quantity;
            }
        }
        pos
    }

    fn latest_delta(&self) -> Option<f64> {
        if self.bitmex_orders.len() == self.deribit_orders.len() {
            if let (Some(last_bitmex_order), Some(last_deribit_order)) =
                (self.bitmex_orders.last(), self.deribit_orders.last())
            {
                if let (Some(last_bitmex_ou), Some(last_deribit_ou)) = (
                    last_bitmex_order.updates.last(),
                    last_deribit_order.updates.last(),
                ) {
                    if (last_bitmex_ou.status == OrderStatus::Filled
                        && last_deribit_ou.status == OrderStatus::Filled)
                        || (last_bitmex_ou.status == OrderStatus::Submitted
                            && last_deribit_ou.status == OrderStatus::Submitted)
                    {
                        return Some(
                            last_bitmex_ou.price.ln()
                                - COINT_COEFF * last_deribit_ou.price.ln()
                                - EQUIL_VAL,
                        );
                    }
                }
            }
        }
        None
    }
}
