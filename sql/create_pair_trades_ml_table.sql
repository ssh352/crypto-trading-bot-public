CREATE TABLE pair_trades_ml (
	id SERIAL NOT NULL PRIMARY KEY,    
    bitmex_order_id TEXT NOT NULL,
    deribit_order_id TEXT NOT NULL,
    quantity INTEGER NOT NULL,
    recent_prices JSON NOT NULL,
    order_book_snapshots JSON NOT NULL,
    recent_trades JSON NOT NULL,
    features JSON NOT NULL,
    submit_timestamp TIMESTAMPTZ NOT NULL,
    is_filled BOOLEAN DEFAULT FALSE NOT NULL,
    bitmex_fill_timestamp TIMESTAMPTZ NULL,
    deribit_fill_timestamp TIMESTAMPTZ NULL,
    max_long_delta_before_first_fill DOUBLE PRECISION NULL, -- not null if quantity > 0 and both orders fill
    min_short_delta_before_first_fill DOUBLE PRECISION NULL, -- not null if quantity < 0 and both orders fill
    max_long_delta_after_first_fill_and_before_second_fill DOUBLE PRECISION NULL, -- not null if quantity > 0 and both orders fill
    min_short_delta_after_first_fill_and_before_second_fill DOUBLE PRECISION NULL -- not null if quantity < 0 and both orders fill
);
CREATE INDEX idx_pair_trades_ml_bitmex_order_id on pair_trades_ml(bitmex_order_id);
CREATE INDEX idx_pair_trades_ml_deribit_order_id on pair_trades_ml(deribit_order_id);
CREATE INDEX idx_pair_trades_ml_submit_timestamp on pair_trades_ml(submit_timestamp);