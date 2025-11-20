from fastapi import FastAPI, WebSocket
import json

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    while True:
        landmarks = await websocket.receive_json()
        with open("./landmarks.json", 'w') as f:
            json.dump(landmarks, f, indent=4)