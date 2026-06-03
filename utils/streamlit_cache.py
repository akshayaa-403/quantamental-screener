import streamlit as st
from functools import wraps

def streamlit_cache(ttl=3600):
    def decorator(func):
        @st.cache_data(ttl=ttl)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        return wrapper
    return decorator