CREATE TABLE events (
	id BIGSERIAL NOT NULL PRIMARY KEY,
	json JSON NOT NULL,
	timestamp TIMESTAMPTZ NOT NULL
);
CREATE INDEX idx_events_timestamp on events(timestamp);