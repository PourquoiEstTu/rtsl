import pytest
import json
import asyncio
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

# THESE TEST CASES REQUIRE A MODEL IN THE REPO
from app.main import app
client = TestClient(app)

def test_read_main():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Hello World"}

@pytest.mark.asyncio
async def test_websocket_null_keypoints():
    try:
        with client.websocket_connect("/ws") as websocket:
            for i in range(50):
                websocket.send_json({"model" : "Showcase", "pose" : {"landmarks" : []}, "hand" : {"landmarks" : [], "handedness" : []}})
            data = websocket.receive_json()
            assert data != None
    except WebSocketDisconnect as e:
        assert e.code == 1000

@pytest.mark.asyncio
async def test_websocket_actual_keypoints():
    with open("app/tests/sample_data/50_sent_jsons.json", "r") as file:
        frames_dictionary = json.load(file)
    try:
        with client.websocket_connect("/ws") as websocket:
            for i in range(50):
                websocket.send_json(frames_dictionary[i])
            data = json.json.loads(websocket.receive_json())
            assert data["word"] != ""
    except WebSocketDisconnect as e:
        assert e.code == 1000