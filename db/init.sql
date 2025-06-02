DROP TABLE IF EXISTS predictions;

CREATE TABLE predictions (
    date_time_stamp TIMESTAMP,
    predicted_digit TEXT,
    true_value TEXT
);