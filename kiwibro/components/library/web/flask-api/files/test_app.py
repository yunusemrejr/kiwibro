import pytest

from app import RESOURCE, create_app


@pytest.fixture()
def client(tmp_path):
    app = create_app(str(tmp_path / "test.db"))
    app.config.update(TESTING=True)
    return app.test_client()


def make_item(client, title="First item"):
    response = client.post(f"/{RESOURCE}", json={"title": title, "body": "Some body"})
    assert response.status_code == 201
    return response.get_json()


def test_health(client):
    body = client.get("/health").get_json()
    assert body["status"] == "ok"


def test_create_and_list(client):
    item = make_item(client)
    assert item["title"] == "First item"
    assert item["archived"] is False
    listed = client.get(f"/{RESOURCE}").get_json()
    assert [entry["id"] for entry in listed] == [item["id"]]


def test_rejects_empty_title(client):
    response = client.post(f"/{RESOURCE}", json={"title": "   "})
    assert response.status_code == 400
    assert "required" in response.get_json()["error"]


def test_rejects_unknown_field(client):
    response = client.post(f"/{RESOURCE}", json={"title": "A", "owner": "B"})
    assert response.status_code == 400
    assert "Unknown field" in response.get_json()["error"]


def test_archive_hides_item_from_default_list(client):
    item = make_item(client)
    response = client.patch(f"/{RESOURCE}/{item['id']}", json={"archived": True})
    assert response.status_code == 200
    assert response.get_json()["archived"] is True
    assert client.get(f"/{RESOURCE}").get_json() == []
    listed = client.get(f"/{RESOURCE}?include_archived=1").get_json()
    assert len(listed) == 1


def test_missing_item_returns_404(client):
    assert client.get(f"/{RESOURCE}/999").status_code == 404
    assert client.patch(f"/{RESOURCE}/999", json={"title": "A"}).status_code == 404


def test_delete_is_not_offered(client):
    item = make_item(client)
    assert client.delete(f"/{RESOURCE}/{item['id']}").status_code == 405
