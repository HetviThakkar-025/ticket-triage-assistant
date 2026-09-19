"""Ticket ownership: one user's token must never reach another user's data."""

from tests.conftest import auth_header, register_and_login

ALICE_MESSAGE = "My order arrived damaged yesterday and it cost 3500 rupees."
BOB_MESSAGE = "Bob's confidential parcel never arrived after 9 days."


def _create_ticket(client, token: str, message: str) -> dict:
    response = client.post("/tickets", json={"message": message}, headers=auth_header(token))
    assert response.status_code == 201, response.text
    return response.json()


def test_alice_cannot_read_bobs_ticket_by_id(client):
    """The headline authorization guarantee."""
    alice_token = register_and_login(client, "alice@example.com")
    bob_token = register_and_login(client, "bob@example.com")

    bob_ticket = _create_ticket(client, bob_token, BOB_MESSAGE)
    bob_ticket_id = bob_ticket["id"]

    # Bob can read his own ticket.
    owner_response = client.get(f"/tickets/{bob_ticket_id}", headers=auth_header(bob_token))
    assert owner_response.status_code == 200
    assert owner_response.json()["message"] == BOB_MESSAGE

    # Alice, with a perfectly valid token of her own, cannot.
    intruder_response = client.get(
        f"/tickets/{bob_ticket_id}", headers=auth_header(alice_token)
    )
    assert intruder_response.status_code == 404
    assert "confidential" not in intruder_response.text
    assert BOB_MESSAGE not in intruder_response.text


def test_ticket_list_is_scoped_to_the_authenticated_user(client):
    alice_token = register_and_login(client, "alice@example.com")
    bob_token = register_and_login(client, "bob@example.com")

    alice_ticket = _create_ticket(client, alice_token, ALICE_MESSAGE)
    _create_ticket(client, bob_token, BOB_MESSAGE)

    response = client.get("/tickets", headers=auth_header(alice_token))
    assert response.status_code == 200
    tickets = response.json()
    assert [t["id"] for t in tickets] == [alice_ticket["id"]]
    assert BOB_MESSAGE not in response.text


def test_unknown_ticket_id_is_404(client):
    token = register_and_login(client, "alice@example.com")
    assert client.get("/tickets/9999", headers=auth_header(token)).status_code == 404


def test_ticket_endpoints_require_authentication(client):
    token = register_and_login(client, "alice@example.com")
    ticket = _create_ticket(client, token, ALICE_MESSAGE)

    assert client.get("/tickets").status_code == 401
    assert client.get(f"/tickets/{ticket['id']}").status_code == 401
    assert client.post("/tickets", json={"message": ALICE_MESSAGE}).status_code == 401


def test_ticket_response_contains_the_decision(client):
    token = register_and_login(client, "alice@example.com")
    ticket = _create_ticket(client, token, ALICE_MESSAGE)
    for key in ("id", "message", "created_at", "action", "confidence", "reason", "sources"):
        assert key in ticket
    assert 0.0 <= ticket["confidence"] <= 1.0
