# RTSL

Real Time Sign Language: An app to translate sign language in real time.

All documents/poster/video is located under misc. 

![Mobile Example Gif](misc/media/mobile_example.gif)

![Extension Example Gif](misc/media/extension_example.gif)

**Try it out yourself at** https://rtsl.cas.mcmaster.ca/ (requires a connection to McMaster wifi).

This app displays word-by-word translations at the top of the screen. Once the signer ends their sentence, the pause is detected an the translated sentence is displayed.

We provide a website, app (PWA), and Chrome extension.

## Run Frontend Locally

Requires `npm`.

```bash
cd src/frontend
npm i
npm run dev
```

For the web extension, see the Releases.

## Run Backend Locally (HTTP)

Requires `uvicorn`. 
Must alter websocket address in `src/frontend/src/components/KeypointTransceiver.vue` (Comment out line 20 and uncomment line 21). 
Must download and overwrite `src/backend/models` with `models` folder under releases.

```bash
cd src/backend
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

This model was trained by ASL Citizen, which was compiled by Microsoft.