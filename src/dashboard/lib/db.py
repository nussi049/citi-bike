"""
db.py - Database Connection Management for Streamlit Dashboard

Provides a cached DuckDB connection for efficient data access across
dashboard pages. Using Streamlit's cache_resource ensures a single
connection is shared across all users and reruns.

Configuration:
    - threads=4: Parallel query execution
    - memory_limit=2GB: Prevents OOM on large queries

Usage:
    from lib.db import get_con
    con = get_con()
    df = con.execute("SELECT * FROM ...").fetch_df()
"""

import duckdb
import streamlit as st


@st.cache_resource
def get_con() -> duckdb.DuckDBPyConnection:
    """
    Get a cached DuckDB connection.

    The connection is created once and reused across all dashboard pages
    and user sessions. This avoids connection overhead on each page load.

    Returns:
        DuckDB connection configured for dashboard use
    """
    con = duckdb.connect(database=":memory:")
    con.execute("PRAGMA threads=4")
    con.execute("PRAGMA memory_limit='2GB'")
    return con
