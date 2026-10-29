from fastapi.testclient import TestClient

from secondbrain.api.main import app

client = TestClient(app)


def test_health():
    assert client.get("/health").json() == {"status": "ok"}


def test_upload_rejects_unknown_extension():
    r = client.post("/documents", files={"file": ("evil.exe", b"x")})
    assert r.status_code == 415
