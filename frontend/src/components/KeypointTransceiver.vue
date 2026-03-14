<script setup lang="ts">
import { onBeforeUnmount, onMounted, ref } from "vue";
import {
  FilesetResolver,
  HandLandmarker,
  PoseLandmarker,
  PoseLandmarkerResult,
  type HandLandmarkerResult,
} from "@mediapipe/tasks-vision";
import { useWebSocket } from "@vueuse/core";

// Check if we are running inside a Chrome Extension
const isExtension = typeof chrome !== 'undefined' && !!chrome.runtime?.id;

let handLandmarker: HandLandmarker | null = null;
let poseLandmarker: PoseLandmarker | null = null;
let animationFrameId: number | null = null;
let stream: MediaStream | null = null;

const videoEl = ref<HTMLVideoElement | null>(null);
const canvasEl = ref<HTMLCanvasElement | null>(null);

// Todo: use env vars for server url
const { send } = useWebSocket(
  "wss://130.113.255.255/ws"
);

onBeforeUnmount(() => {
  // Close MediaPipe
  handLandmarker?.close();
  poseLandmarker?.close();

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

    const landmarkers = await initMediaPipe();
    handLandmarker = landmarkers.handLandmarker
    poseLandmarker = landmarkers.poseLandmarker

    if (!handLandmarker) {
      console.error("Failed to initialize Hand Landmarker");
      return;
    }
    if (!poseLandmarker) {
      console.error("Failed to initialize Pose Landmarker");
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
    let handLandmarkerResults: HandLandmarkerResult | undefined = undefined;
    let poseLandmarkerResults: PoseLandmarkerResult | undefined = undefined;

    const predictWebcam = () => {
      try {
        if (!videoEl.value || !canvasEl.value || !handLandmarker || !poseLandmarker) return;

        const videoWidth = videoEl.value.videoWidth;
        const videoHeight = videoEl.value.videoHeight;

        // Set canvas to video's intrinsic size
        canvasEl.value.width = videoWidth;
        canvasEl.value.height = videoHeight;

        const startTimeMs = performance.now();
        if (lastVideoTime !== videoEl.value.currentTime) {
          lastVideoTime = videoEl.value.currentTime;
          handLandmarkerResults = handLandmarker.detectForVideo(videoEl.value, startTimeMs);
          poseLandmarkerResults = poseLandmarker.detectForVideo(videoEl.value, startTimeMs);
        }

        canvasCtx.clearRect(0, 0, canvasEl.value.width, canvasEl.value.height);

        // todo sending pose
        if (handLandmarkerResults?.landmarks && poseLandmarkerResults?.landmarks) {
          drawLandmarks(
            canvasCtx,
            handLandmarkerResults,
            poseLandmarkerResults,
            canvasEl.value.width,
            canvasEl.value.height
          );

          // Combine results and send over websocket
          send(JSON.stringify({ hand: handLandmarkerResults, pose: poseLandmarkerResults }));
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

// Takes a moment to get everything needed to run.
async function initMediaPipe() {
  // Use extension URL if available, otherwise fallback to your current web app path
  const wasmPath = isExtension ? chrome.runtime.getURL("/wasm") : "/wasm";
  const modelPath = (landmarkerType: 'hand' | 'pose') => isExtension
    ? chrome.runtime.getURL("/models/hand_landmarker.task")
    : `/models/${landmarkerType}_landmarker.task`;

  const vision = await FilesetResolver.forVisionTasks(wasmPath);

  return {
    handLandmarker: await HandLandmarker.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath: modelPath('hand'),
        delegate: "GPU",
      },
      runningMode: "VIDEO",
      numHands: 2,
    }),
    poseLandmarker: await PoseLandmarker.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath: modelPath('pose'),
        delegate: "GPU",
      },
      runningMode: "VIDEO",
    })
  };
}


function drawLandmarks(
  ctx: CanvasRenderingContext2D,
  handLandmarkerResults: HandLandmarkerResult,
  poseLandmarkerResults: PoseLandmarkerResult,
  W: number,
  H: number
) {
  for (const landmarks of handLandmarkerResults.landmarks) {
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

  for (const landmarks of poseLandmarkerResults.landmarks) {
    // Draw all 21 landmarks as circles
    for (const landmark of landmarks) {
      const x = landmark.x * W;
      const y = landmark.y * H;

      ctx.beginPath();
      ctx.arc(x, y, 5, 0, 2 * Math.PI);
      ctx.fillStyle = "#007FFF";
      ctx.fill();
    }

    // Draw connections between landmarks
    const connections = PoseLandmarker.POSE_CONNECTIONS;
    ctx.strokeStyle = "#007FFF";
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