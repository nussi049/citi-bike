import duckdb
import streamlit as st

@st.cache_resource
def get_con():
    con = duckdb.connect(database=":memory:")
    con.execute("PRAGMA threads=4")
    con.execute("PRAGMA memory_limit='2GB'")
    return con
