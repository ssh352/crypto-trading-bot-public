#![feature(map_first_last)]
#![feature(or_patterns)]
#![allow(clippy::float_cmp)] // strict float equality comparisons
#![allow(clippy::new_without_default)]
#![allow(clippy::type_complexity)]
#![allow(clippy::collapsible_else_if)]
#![allow(clippy::comparison_chain)]

use serde::{Deserialize, Serialize};
use std::fmt::Display;

pub mod algo;
pub mod api_response;
pub mod client;
pub mod constants;
pub mod event;
pub mod exchange;
pub mod features;
pub mod ml;
pub mod order;
pub mod order_book;
pub mod order_book_snapshots;
pub mod portfolio;
pub mod price;

#[derive(Serialize, Deserialize, Eq, PartialEq, Copy, Clone, Debug)]
pub enum Exchange {
    Bitmex,
    Deribit,
}

impl Display for Exchange {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Exchange::Bitmex => {
                write!(f, "Bitmex")
            }
            Exchange::Deribit => {
                write!(f, "Deribit")
            }
        }
    }
}
