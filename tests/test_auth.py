"""Registration, login, and token validation."""

import jwt
import pytest

from src.auth import hash_password, verify_password
from tests.conftest import auth_header, register_and_login


def test_password_is_hashed_not_stored_in_plaintext():
    hashed = hash_password("password123")
    assert hashed != "password123"
    assert hashed.startswith("$2")  # bcrypt
    assert verify_password("password123", hashed)
    assert not verify_password("wrong-password", hashed)


def test_register_returns_user_without_password(client):
    response = client.post(
        "/register", json={"email": "alice@example.com", "password": "password123"}
    )
    assert response.status_code == 201
    body = response.json()
    assert body["email"] == "alice@example.com"
    assert "password" not in body and "password_hash" not in body


def test_register_rejects_duplicate_email(client):
    payload = {"email": "alice@example.com", "password": "password123"}
    assert client.post("/register", json=payload).status_code == 201
    assert client.post("/register", json=payload).status_code == 409


def test_register_rejects_short_password(client):
    response = client.post("/register", json={"email": "a@example.com", "password": "short"})
    assert response.status_code == 422


def test_login_returns_usable_jwt(client):
    token = register_and_login(client, "alice@example.com")
    claims = jwt.decode(token, options={"verify_signature": False})
    assert claims["email"] == "alice@example.com"
    assert claims["exp"] > claims["iat"]

    response = client.get("/me", headers=auth_header(token))
    assert response.status_code == 200
    assert response.json()["email"] == "alice@example.com"


def test_login_with_wrong_password_is_rejected(client):
    client.post("/register", json={"email": "alice@example.com", "password": "password123"})
    response = client.post(
        "/login", json={"email": "alice@example.com", "password": "not-the-password"}
    )
    assert response.status_code == 401
    assert "access_token" not in response.json()


def test_login_with_unknown_email_is_rejected(client):
    response = client.post(
        "/login", json={"email": "nobody@example.com", "password": "password123"}
    )
    assert response.status_code == 401


@pytest.mark.parametrize(
    "headers",
    [{}, {"Authorization": "Bearer not-a-real-token"}, {"Authorization": "Basic abc"}],
)
def test_protected_route_requires_valid_bearer_token(client, headers):
    assert client.get("/me", headers=headers).status_code == 401


def test_token_signed_with_another_secret_is_rejected(client):
    forged = jwt.encode({"sub": "1", "exp": 9999999999}, "attacker-secret", algorithm="HS256")
    assert client.get("/me", headers=auth_header(forged)).status_code == 401
