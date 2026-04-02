-- Initial schema for ETL demo

CREATE TABLE IF NOT EXISTS customers (
    id         BIGINT PRIMARY KEY,
    name       TEXT        NOT NULL,
    email      TEXT        NOT NULL,
    age        FLOAT,
    country    VARCHAR(3),
    revenue    FLOAT       DEFAULT 0
);
