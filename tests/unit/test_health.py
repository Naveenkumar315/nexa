from fastapi.testclient import TestClient

from app.main import create_app


def test_health_returns_ok() -> None:
    with TestClient(create_app()) as client:  # `with` runs the lifespan
        response = client.get("/api/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["app"] == "Nexa"
    assert body["sample_rate"] == 16000
