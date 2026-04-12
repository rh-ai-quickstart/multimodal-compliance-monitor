"""Minimal database helpers for eval seed loading."""

import os
from contextlib import contextmanager

import psycopg2

DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "ppe_tracking")
DB_USER = os.getenv("DB_USER", "ppe_user")
DB_PASSWORD = os.getenv("DB_PASSWORD", "ppe_password")


def _conn_string() -> str:
    return f"host={DB_HOST} port={DB_PORT} dbname={DB_NAME} user={DB_USER} password={DB_PASSWORD}"


@contextmanager
def get_connection():
    conn = psycopg2.connect(_conn_string())
    try:
        yield conn
    finally:
        conn.close()
