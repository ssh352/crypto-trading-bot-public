use serde::{Deserialize, Serialize};
use std::{
    fmt::Display,
    ops::{Add, Sub},
};

#[derive(Default, Debug, Eq, PartialEq, Ord, PartialOrd, Copy, Clone, Serialize, Deserialize)]
pub struct Price {
    pub price_x100: i32,
}

impl Price {
    pub fn from_f32(price: f32) -> Self {
        Price {
            price_x100: (price * 100.) as i32,
        }
    }

    pub fn from_f64(price: f64) -> Self {
        Price {
            price_x100: (price * 100.) as i32,
        }
    }

    pub fn as_f32(&self) -> f32 {
        self.price_x100 as f32 / 100.
    }

    pub fn as_f64(&self) -> f64 {
        self.price_x100 as f64 / 100.
    }

    pub fn ln(&self) -> f64 {
        (self.price_x100 as f64 / 100.).ln()
    }
}

impl Add for Price {
    type Output = Self;

    fn add(self, rhs: Self) -> Self {
        Self {
            price_x100: self.price_x100 + rhs.price_x100,
        }
    }
}

impl Sub for Price {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self {
        Self {
            price_x100: self.price_x100 - rhs.price_x100,
        }
    }
}

impl Display for Price {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", (self.price_x100 as f32 / 100.))
    }
}
