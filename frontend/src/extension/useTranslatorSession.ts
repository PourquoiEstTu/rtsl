import {
  FilesetResolver,
  HandLandmarker,
  type HandLandmarkerResult
} from "@mediapipe/tasks-vision";
import { ref } from "vue";

export function useTranslatorSession() {
    const sessionStatus = ref("Initializing");
    const socketStatus = ref("Closed");
    const translationText = ref("Waiting for backend response...");
    const debugText = ref("");
    const sourceTitle = ref("-");

    let handLandmarker: HandLandmarker | null = null;
    let animationFrameId: number | null = null;
    let capturedStream: MediaStream | null = null;
    let ws: WebSocket | null = null;
    let isRunning = false;
    let lastVideoTime = -1;

    const WS_URL = "ws://127.0.0.1:8000/ws";

    function setDebug(text: string) {
        debugText.value = text;
    }

    function appendDebug(text: string) {
        debugText.value = `${text}\n${debugText.value}`;
    }

    async function initMediaPipe() {
        const vision = await FilesetResolver.forVisionTasks(
            chrome.runtime.getURL("/wasm")
        );

        handLandmarker = await HandLandmarker.createFromOptions(vision, {
            baseOptions: {
                modelAssetPath: chrome.runtime.getURL("/models/hand_landmarker.task"),
                delegate: "GPU"
            },
            runningMode: "VIDEO",
            numHands: 2
        });
    }

//   function connectWebSocket() {
//     if (
//       ws &&
//       (ws.readyState === WebSocket.OPEN ||
//         ws.readyState === WebSocket.CONNECTING)
//     ) {
//       return;
//     }

//     socketStatus.value = "Connecting";
//     ws = new WebSocket(WS_URL);

//     ws.onopen = () => {
//       socketStatus.value = "Open";
//       appendDebug("WebSocket connected.");
//     };

//     ws.onmessage = (event) => {
//       try {
//         const payload = JSON.parse(event.data);
//         if (payload.translation) {
//           translationText.value = payload.translation;
//         } else {
//           translationText.value = event.data;
//         }
//       } catch {
//         translationText.value = event.data;
//       }
//     };

//     ws.onerror = () => {
//       socketStatus.value = "Error";
//       appendDebug("WebSocket error.");
//     };

//     ws.onclose = () => {
//       socketStatus.value = "Closed";
//       appendDebug("WebSocket closed.");
//     };
//   }
    function connectWebSocket() {
        socketStatus.value = "Disabled for local test";
    }

    async function consumeCapturedTabStream(
        streamId: string,
        tabTitle: string,
        videoEl: HTMLVideoElement
    ) {
        if (capturedStream) {
            stopCapture();
        }

        sourceTitle.value = tabTitle || "Meeting Tab";

        capturedStream = await navigator.mediaDevices.getUserMedia({
            audio: false,
            video: {
                mandatory: {
                chromeMediaSource: "tab",
                chromeMediaSourceId: streamId,
                maxWidth: 1920,
                maxHeight: 1080,
                maxFrameRate: 30
                }
            } as MediaTrackConstraints
        });

        videoEl.srcObject = capturedStream;
        await videoEl.play();

        sessionStatus.value = "Captured tab ready";
        appendDebug("Captured tab stream attached.");
    }

    function drawLandmarks(
        ctx: CanvasRenderingContext2D,
        results: HandLandmarkerResult,
        W: number,
        H: number
    ) {
        for (const landmarks of results.landmarks || []) {
            for (const landmark of landmarks) {
                const x = landmark.x * W;
                const y = landmark.y * H;

                ctx.beginPath();
                ctx.arc(x, y, 4, 0, 2 * Math.PI);
                ctx.fillStyle = "#ffff00";
                ctx.fill();
            }

            const connections = HandLandmarker.HAND_CONNECTIONS;
            ctx.strokeStyle = "#ffff00";
            ctx.lineWidth = 2;

            for (const conn of connections) {
                const start = landmarks[conn.start];
                const end = landmarks[conn.end];
                if (!start || !end) continue;

                ctx.beginPath();
                ctx.moveTo(start.x * W, start.y * H);
                ctx.lineTo(end.x * W, end.y * H);
                ctx.stroke();
            }
        }
    }

    function startLoop(
        videoEl: HTMLVideoElement,
        canvasEl: HTMLCanvasElement
    ) {
        if (!handLandmarker || !videoEl.srcObject) {
            appendDebug("Cannot start loop: missing MediaPipe or stream.");
            return;
        }

        if (isRunning) return;

        isRunning = true;
        sessionStatus.value = "Running";
        connectWebSocket();

        const canvasCtx = canvasEl.getContext("2d");
        if (!canvasCtx) {
            appendDebug("No canvas context.");
            return;
        }

        const predict = () => {
            if (!isRunning || !handLandmarker) return;

            try {
                const videoWidth = videoEl.videoWidth || 1280;
                const videoHeight = videoEl.videoHeight || 720;
                canvasEl.width = videoWidth;
                canvasEl.height = videoHeight;

                if (lastVideoTime !== videoEl.currentTime) {
                    lastVideoTime = videoEl.currentTime;
                    const results = handLandmarker.detectForVideo(
                        videoEl,
                        performance.now()
                    );

                    canvasCtx.clearRect(0, 0, canvasEl.width, canvasEl.height);

                    if (results?.landmarks?.length) {
                        drawLandmarks(canvasCtx, results, canvasEl.width, canvasEl.height);

                        if (ws && ws.readyState === WebSocket.OPEN) {
                            ws.send(
                                JSON.stringify({
                                type: "mediapipe_landmarks",
                                payload: results
                                })
                            );
                        }
                    }
                }
            } catch (error) {
                appendDebug(
                    `Prediction loop error: ${
                        error instanceof Error ? error.message : String(error)
                    }`
                );
            }

            animationFrameId = window.requestAnimationFrame(predict);
        };

        animationFrameId = window.requestAnimationFrame(predict);
    }

    function stopLoopOnly() {
        isRunning = false;
        sessionStatus.value = "Stopped";

        if (animationFrameId) {
            cancelAnimationFrame(animationFrameId);
            animationFrameId = null;
        }
    }

    function stopCapture() {
        stopLoopOnly();

        if (capturedStream) {
            capturedStream.getTracks().forEach((track) => track.stop());
            capturedStream = null;
        }

        if (ws) {
            ws.close();
            ws = null;
        }

        sessionStatus.value = "Idle";
    }

    function cleanup() {
        stopCapture();
        handLandmarker?.close();
        handLandmarker = null;
    }

    return {
        sessionStatus,
        socketStatus,
        translationText,
        debugText,
        sourceTitle,
        setDebug,
        appendDebug,
        initMediaPipe,
        consumeCapturedTabStream,
        startLoop,
        stopCapture,
        cleanup
    };
}