import pytest
from fastapi.testclient import TestClient

from app import RESOURCE, create_app


@pytest.fixture()
def client(tmp_path):
    return TestClient(create_app(str(tmp_path / "test.db")))


def make_item(client, title="First item"):
    response = client.post(f"/{RESOURCE}", json={"title": title, "body": "Some body"})
    assert response.status_code == 201
    return response.json()


def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"


def test_create_strips_title_and_lists(client):
    item = make_item(client, "  First item  ")
    assert item["title"] == "First item"
    assert item["archived"] is False
    listed = client.get(f"/{RESOURCE}").json()
    assert [entry["id"] for entry in listed] == [item["id"]]


def test_rejects_blank_title(client):
    response = client.post(f"/{RESOURCE}", json={"title": "   "})
    assert response.status_code == 422


def test_rejects_unknown_field(client):
    response = client.post(f"/{RESOURCE}", json={"title": "A", "owner": "B"})
    assert response.status_code == 422


def test_archive_hides_item_from_default_list(client):
    item = make_item(client)
    response = client.patch(f"/{RESOURCE}/{item['id']}", json={"archived": True})
    assert response.status_code == 200
    assert response.json()["archived"] is True
    assert client.get(f"/{RESOURCE}").json() == []
    listed = client.get(f"/{RESOURCE}?include_archived=true").json()
    assert len(listed) == 1


def test_empty_patch_is_rejected(client):
    item = make_item(client)
    response = client.patch(f"/{RESOURCE}/{item['id']}", json={})
    assert response.status_code == 400


def test_missing_item_returns_404(client):
    assert client.get(f"/{RESOURCE}/999").status_code == 404
    assert client.patch(f"/{RESOURCE}/999", json={"title": "A"}).status_code == 404


def test_delete_is_not_offered(client):
    item = make_item(client)
    assert client.delete(f"/{RESOURCE}/{item['id']}").status_code == 405
