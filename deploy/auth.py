"""
Simple password protection for Streamlit dashboard.

To enable:
1. Set environment variable: export DASHBOARD_PASSWORD="your_password"
2. Or create .streamlit/secrets.toml with: password = "your_password"

To disable:
- Don't set the environment variable or secret
"""
# =============================================================================
# PASSWORD PROTECTION
# Set DASHBOARD_PASSWORD environment variable to enable
#
# NOTE: This is a simple protection for the coding challenge demo.
# NOT brute-force resistant - no rate limiting, no account lockout.
# For production use: implement proper auth (OAuth, SSO, etc.)
# =============================================================================

import os
import streamlit as st


def check_password() -> bool:
    """
    Returns True if the user has entered the correct password or no password is set.
    """
    # Get password from environment or secrets
    password = os.environ.get("DASHBOARD_PASSWORD")

    if not password:
        try:
            password = st.secrets.get("password")
        except Exception:
            pass

    # If no password configured, allow access
    if not password:
        return True

    # Initialize session state
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False

    # Already authenticated
    if st.session_state.authenticated:
        return True

    # Show login form
    st.title("NYC Bike Crash Dashboard")
    st.markdown("---")

    with st.form("login_form"):
        entered_password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Login")

        if submitted:
            if entered_password == password:
                st.session_state.authenticated = True
                st.rerun()
            else:
                st.error("Incorrect password")

    st.markdown("---")
    st.caption("Contact: [Your Email]")

    return False
