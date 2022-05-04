from server import create_app
import pytest

@pytest.fixture()
def app():
    app = create_app()

    app.config.update({
        "TESTING": True,
    })

    yield app

@pytest.fixture()
def client(app):
    return app.test_client()

def test_respond(client):
    response = client.post("/respond", json={"text": "Hello there"})
    print(response)
    print(response.get_json())
    assert response.status_code == 200
    assert response.get_json()['prediction'][0] == 'greet'
