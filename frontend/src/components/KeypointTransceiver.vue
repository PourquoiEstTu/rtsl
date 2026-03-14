<script setup lang="ts">
import { onBeforeUnmount, onMounted, ref } from "vue";
import {
  FilesetResolver,
  HandLandmarker,
  type HandLandmarkerResult,
} from "@mediapipe/tasks-vision";
import { useWebSocket } from "@vueuse/core";

// Check if we are running inside a Chrome Extension
const isExtension = typeof chrome !== 'undefined' && !!chrome.runtime?.id;

let handLandmarker: HandLandmarker | null = null;
let animationFrameId: number | null = null;
let stream: MediaStream | null = null;

const videoEl = ref<HTMLVideoElement | null>(null);
const canvasEl = ref<HTMLCanvasElement | null>(null);

// Todo: use env vars for server url
const { send } = useWebSocket(
  "http://127.0.0.1:8000/ws"
);

onBeforeUnmount(() => {
  // Close MediaPipe
  handLandmarker?.close();

  // Stop camera stream
  if (stream) {
    stream.getTracks().forEach((track) => track.stop());
  }

  // Cancel animation frame
  if (animationFrameId) {
    cancelAnimationFrame(animationFrameId);
  }
});

onMounted(async () => {
  try {
    // If these are null, the template does not have refs to the needed elements
    if (!videoEl.value || !canvasEl.value) {
      console.error("Missing references to video or canvas elements");
      return;
    }

    handLandmarker = await initMediaPipe();
    if (!handLandmarker) {
      console.error("Failed to initialize Hand Landmarker");
      return;
    }

    const canvasCtx = canvasEl.value.getContext("2d");
    if (!canvasCtx) {
      console.error("Failed to get canvas context");
      return;
    }

    stream = isExtension
      ? await navigator.mediaDevices.getDisplayMedia({ video: true })                        // If its an extension, use screen capture
      : await navigator.mediaDevices.getUserMedia({ video: { facingMode: "environment" } }); // Otherwise (app), use camera
    videoEl.value.srcObject = stream;

    let lastVideoTime = -1;
    let results: HandLandmarkerResult | undefined = undefined;

    const predictWebcam = () => {
      try {
        if (!videoEl.value || !canvasEl.value || !handLandmarker) return;

        const videoWidth = videoEl.value.videoWidth;
        const videoHeight = videoEl.value.videoHeight;

        // Set canvas to video's intrinsic size
        canvasEl.value.width = videoWidth;
        canvasEl.value.height = videoHeight;

        const startTimeMs = performance.now();
        if (lastVideoTime !== videoEl.value.currentTime) {
          lastVideoTime = videoEl.value.currentTime;
          results = handLandmarker.detectForVideo(videoEl.value, startTimeMs);
        }

        canvasCtx.clearRect(0, 0, canvasEl.value.width, canvasEl.value.height);

        if (results?.landmarks) {
          drawLandmarks(
            canvasCtx,
            results,
            canvasEl.value.width,
            canvasEl.value.height
          );
          // Send results via WebSocket once per frame
          send(JSON.stringify(results));
        }
      } catch (error) {
        console.error("Error in prediction loop:", error);
      }

      animationFrameId = window.requestAnimationFrame(predictWebcam);
    };

    // Start prediction when video is ready
    videoEl.value.addEventListener("loadeddata", predictWebcam);
  } catch (error) {
    console.error("Failed to initialize camera or MediaPipe:", error);
    // Todo: show user-friendly error in UI
    // e.g., "Camera access denied" or "Failed to load hand tracking model"
  }
});

// Before we can use HandLandmarker class we must wait for it to finish
// loading. Machine Learning models can be large and take a moment to
// get everything needed to run.
async function initMediaPipe() {
  // Use extension URL if available, otherwise fallback to your current web app path
  const wasmPath = isExtension ? chrome.runtime.getURL("/wasm") : "/wasm";
  const modelPath = isExtension
    ? chrome.runtime.getURL("/models/hand_landmarker.task")
    : "/models/hand_landmarker.task";

  const vision = await FilesetResolver.forVisionTasks(wasmPath);
  return await HandLandmarker.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath: modelPath,
      delegate: "GPU",
    },
    runningMode: "VIDEO",
    numHands: 2,
  });
}


function drawLandmarks(
  ctx: CanvasRenderingContext2D,
  results: HandLandmarkerResult,
  W: number,
  H: number
) {
  for (const landmarks of results.landmarks) {
    // Draw all 21 landmarks as circles
    for (const landmark of landmarks) {
      const x = landmark.x * W;
      const y = landmark.y * H;

      ctx.beginPath();
      ctx.arc(x, y, 5, 0, 2 * Math.PI);
      ctx.fillStyle = "#FFFF00";
      ctx.fill();
    }

    // Draw connections between landmarks
    const connections = HandLandmarker.HAND_CONNECTIONS;
    ctx.strokeStyle = "#FFFF00";
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
</script>

<template>
  <video v-show="!isExtension" ref="videoEl" autoplay playsinline />
  <canvas v-show="!isExtension" ref="canvasEl" class="absolute w-full object-cover h-full" />
</template>