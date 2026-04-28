import os
import requests
import streamlit as st

API_BASE_URL = os.getenv("API_URL", "http://localhost:8000/api")

# Default timeout (seconds) for all API calls
# Bug 10 Fix: every request now has a timeout so the UI never hangs indefinitely
_TIMEOUT = 10


class APIClient:

    @staticmethod
    def _auth_headers() -> dict:
        """Returns Authorization header if a JWT token exists in session state."""
        token = st.session_state.get("auth_token", "")
        if token:
            return {"Authorization": f"Bearer {token}"}
        return {}

    @staticmethod
    def ping_server() -> bool:
        """Checks if the FastAPI backend is online."""
        try:
            response = requests.get(f"{API_BASE_URL}/health", timeout=3)
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False

    @staticmethod
    def signup(email: str, password: str):
        return requests.post(
            f"{API_BASE_URL}/signup",
            json={"email": email, "password": password},
            timeout=_TIMEOUT,
        )

    @staticmethod
    def login(email: str, password: str):
        return requests.post(
            f"{API_BASE_URL}/login",
            json={"email": email, "password": password},
            timeout=_TIMEOUT,
        )

    @staticmethod
    def reset_password(email: str, new_password: str):
        return requests.post(
            f"{API_BASE_URL}/forgot-password",
            json={"email": email, "new_password": new_password},
            timeout=_TIMEOUT,
        )

    @staticmethod
    def get_history(email: str):
        response = requests.get(
            f"{API_BASE_URL}/user-history/{email}",
            headers=APIClient._auth_headers(),
            timeout=_TIMEOUT,
        )
        response.raise_for_status()
        return response.json()

    @staticmethod
    def save_record(payload: dict):
        return requests.post(
            f"{API_BASE_URL}/save-record",
            json=payload,
            headers=APIClient._auth_headers(),
            timeout=_TIMEOUT,
        )