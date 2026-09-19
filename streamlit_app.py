"""Streamlit frontend for the Ticket Triage Assistant.

This app talks to the FastAPI backend over HTTP only -- it never imports the
database, the retrieval index, or the decision pipeline.
"""

from __future__ import annotations

import os
from typing import Any, Optional

import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
TIMEOUT = 120

ACTION_TONE = {
    "APPROVE_RETURN": "success",
    "APPROVE_REFUND_OR_REPLACEMENT": "success",
    "APPROVE_REPLACEMENT": "success",
    "REPLACE_CORRECT_ITEM": "success",
    "CANCEL_AND_REFUND": "success",
    "OFFER_REPLACEMENT_OR_REFUND": "success",
    "REQUEST_PHOTOS": "warning",
    "REQUEST_DEFECT_EVIDENCE": "warning",
    "WAIT_AND_TRACK": "warning",
    "OPEN_SHIPPING_INVESTIGATION": "warning",
    "NEEDS_MORE_INFORMATION": "warning",
    "REJECT_OPENED_ITEM": "error",
    "REJECT_FOOD_RETURN": "error",
    "REJECT_OUTSIDE_WINDOW": "error",
    "CANNOT_CANCEL_AFTER_DISPATCH": "error",
}


# --- API helpers --------------------------------------------------------

def api_request(
    method: str, path: str, *, token: Optional[str] = None, json: Any = None
) -> tuple[bool, Any]:
    """Return (ok, payload-or-error-message). Never raises."""
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    try:
        response = requests.request(
            method, f"{API_BASE_URL}{path}", headers=headers, json=json, timeout=TIMEOUT
        )
    except requests.RequestException as exc:
        return False, f"Could not reach the API at {API_BASE_URL} ({exc})."

    if response.status_code == 401:
        return False, "Your session has expired. Please log in again."
    if not response.ok:
        try:
            detail = response.json().get("detail", response.text)
        except ValueError:
            detail = response.text
        if isinstance(detail, list) and detail:  # pydantic validation errors
            detail = detail[0].get("msg", str(detail))
        return False, str(detail)
    return True, response.json() if response.content else None


def logout() -> None:
    for key in ("token", "user", "selected_ticket_id", "last_decision"):
        st.session_state.pop(key, None)


# --- rendering ----------------------------------------------------------

def render_decision(ticket: dict) -> None:
    action = ticket["action"]
    banner = {"success": st.success, "error": st.error}.get(
        ACTION_TONE.get(action, "warning"), st.warning
    )
    banner(f"**{action.replace('_', ' ').title()}**  ·  `{action}`")

    st.progress(
        min(max(float(ticket["confidence"]), 0.0), 1.0),
        text=f"Confidence: {float(ticket['confidence']):.0%}",
    )
    st.markdown("**Reason**")
    st.write(ticket["reason"])

    st.markdown("**Policy sources**")
    if ticket["sources"]:
        st.markdown(" ".join(f"`{source}`" for source in ticket["sources"]))
    else:
        st.caption("No policy source was cited.")


# --- views --------------------------------------------------------------

def view_login() -> None:
    st.subheader("Sign in")
    login_tab, register_tab = st.tabs(["Log in", "Register"])

    with login_tab:
        with st.form("login_form"):
            email = st.text_input("Email", key="login_email")
            password = st.text_input("Password", type="password", key="login_password")
            submitted = st.form_submit_button("Log in", use_container_width=True)
        if submitted:
            ok, payload = api_request(
                "POST", "/login", json={"email": email, "password": password}
            )
            if not ok:
                st.error(payload)
            else:
                st.session_state["token"] = payload["access_token"]
                ok, user = api_request("GET", "/me", token=payload["access_token"])
                st.session_state["user"] = user if ok else {"email": email}
                st.rerun()

    with register_tab:
        with st.form("register_form"):
            email = st.text_input("Email", key="register_email")
            password = st.text_input(
                "Password", type="password", key="register_password",
                help="At least 8 characters.",
            )
            submitted = st.form_submit_button("Create account", use_container_width=True)
        if submitted:
            ok, payload = api_request(
                "POST", "/register", json={"email": email, "password": password}
            )
            if not ok:
                st.error(payload)
            else:
                st.success("Account created. You can log in now.")


def view_new_decision(token: str) -> None:
    st.subheader("New decision")
    st.caption(
        "Describe the customer's issue in plain language. The assistant retrieves the "
        "relevant policy rules and decides the next action."
    )

    with st.form("ticket_form"):
        message = st.text_area(
            "Ticket message",
            height=140,
            placeholder="My ₹3,500 order arrived damaged yesterday.",
        )
        submitted = st.form_submit_button("Get decision", use_container_width=True)

    if submitted:
        if not message.strip():
            st.warning("Please enter the ticket message.")
            return
        with st.spinner("Retrieving policy and deciding..."):
            ok, payload = api_request(
                "POST", "/tickets", token=token, json={"message": message.strip()}
            )
        if not ok:
            st.error(payload)
            if "session has expired" in str(payload):
                logout()
        else:
            st.session_state["last_decision"] = payload

    if st.session_state.get("last_decision"):
        st.divider()
        render_decision(st.session_state["last_decision"])


def view_history(token: str) -> None:
    st.subheader("History")
    ok, tickets = api_request("GET", "/tickets", token=token)
    if not ok:
        st.error(tickets)
        if "session has expired" in str(tickets):
            logout()
        return

    if not tickets:
        st.info("No tickets yet. Submit one from the **New Decision** view.")
        return

    st.caption(f"{len(tickets)} ticket(s). Select one to see the full decision.")
    left, right = st.columns([1, 1.4], gap="large")

    with left:
        for ticket in tickets:
            preview = ticket["message"][:60] + ("..." if len(ticket["message"]) > 60 else "")
            if st.button(
                f"#{ticket['id']} · {ticket['action']}\n\n{preview}",
                key=f"ticket-{ticket['id']}",
                use_container_width=True,
            ):
                st.session_state["selected_ticket_id"] = ticket["id"]

    with right:
        selected_id = st.session_state.get("selected_ticket_id")
        if selected_id is None:
            st.info("Select a ticket on the left.")
            return
        # Detail always comes from GET /tickets/{id}, which enforces ownership.
        ok, ticket = api_request("GET", f"/tickets/{selected_id}", token=token)
        if not ok:
            st.error(ticket)
            return
        st.markdown(f"**Ticket #{ticket['id']}** · {ticket['created_at']}")
        st.markdown("**Message**")
        st.write(ticket["message"])
        st.divider()
        render_decision(ticket)


def main() -> None:
    st.set_page_config(page_title="Ticket Triage Assistant", page_icon="🎫", layout="wide")
    st.title("🎫 Ticket Triage Assistant")

    token = st.session_state.get("token")

    with st.sidebar:
        st.markdown("### Session")
        st.caption(f"API: `{API_BASE_URL}`")
        if token:
            user = st.session_state.get("user") or {}
            st.success(f"Signed in as **{user.get('email', 'unknown')}**")
            view = st.radio("View", ["New Decision", "History"], label_visibility="collapsed")
            if st.button("Log out", use_container_width=True):
                logout()
                st.rerun()
        else:
            st.info("Not signed in.")
            view = "Login"

    if not token:
        view_login()
    elif view == "New Decision":
        view_new_decision(token)
    else:
        view_history(token)


if __name__ == "__main__":
    main()
