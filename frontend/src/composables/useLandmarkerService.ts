import {
  FilesetResolver,
  HandLandmarker,
  PoseLandmarker,
  type PoseLandmarkerResult,
  type HandLandmarkerResult,
} from "@mediapipe/tasks-vision";
import { ref, type Ref } from "vue";

let handLandmarker: HandLandmarker | null = null;
let poseLandmarker: PoseLandmarker | null = null;
let animationFrameId = ref<number | null>(null);
let stream: MediaStream | null = null;
let lastVideoTime = -1;

export default function useLandmarkerService() {
  // Takes a moment to get everything needed to run.
  async function init(videoEl: Ref<HTMLVideoElement | null>, isExtension: boolean) {
    // Use extension URL if available, otherwise fallback to your current web app path
    const wasmPath = isExtension ? chrome.runtime.getURL("/wasm") : "/wasm";
    const modelPath = (landmarkerType: "hand" | "pose") =>
      isExtension ? chrome.runtime.getURL("/models/hand_landmarker.task") : `/models/${landmarkerType}_landmarker.task`;

    const vision = await FilesetResolver.forVisionTasks(wasmPath);

    handLandmarker = await HandLandmarker.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath: modelPath("hand"),
        delegate: "GPU",
      },
      runningMode: "VIDEO",
      numHands: 2,
    });
    if (!handLandmarker) {
      console.error("Failed to initialize Hand Landmarker");
      return;
    }

    poseLandmarker = await PoseLandmarker.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath: modelPath("pose"),
        delegate: "GPU",
      },
      runningMode: "VIDEO",
    });
    if (!poseLandmarker) {
      console.error("Failed to initialize Pose Landmarker");
      return;
    }

    stream = isExtension
      ? await navigator.mediaDevices.getDisplayMedia({ video: true }) // If its an extension, use screen capture
      : await navigator.mediaDevices.getUserMedia({ video: { facingMode: "environment" } }); // Otherwise (app), use camera

    if (videoEl.value) videoEl.value.srcObject = stream;
  }

  function stop() {
    // Close MediaPipe
    handLandmarker?.close();
    poseLandmarker?.close();

    // Stop camera stream
    if (stream) {
      stream.getTracks().forEach((track) => track.stop());
    }

    // Cancel animation frame
    if (animationFrameId.value) {
      cancelAnimationFrame(animationFrameId.value);
    }
  }

  function getLandmarks(videoEl: Ref<HTMLVideoElement | null>, canvasEl: Ref<HTMLCanvasElement | null>) {
    let handLandmarkerResults: HandLandmarkerResult | undefined = undefined;
    let poseLandmarkerResults: PoseLandmarkerResult | undefined = undefined;

    if (!videoEl.value || !canvasEl.value || !handLandmarker || !poseLandmarker) {
      return { handLandmarkerResults, poseLandmarkerResults };
    }

    const startTimeMs = performance.now();
    if (lastVideoTime !== videoEl.value.currentTime) {
      lastVideoTime = videoEl.value.currentTime;
      handLandmarkerResults = handLandmarker.detectForVideo(videoEl.value, startTimeMs);
      poseLandmarkerResults = poseLandmarker.detectForVideo(videoEl.value, startTimeMs);
    }

    return { handLandmarkerResults, poseLandmarkerResults };
  }

  async function getVideoSrc(isExtension: boolean): Promise<MediaStream> {
    stream = isExtension
      ? await navigator.mediaDevices.getDisplayMedia({ video: true }) // If its an extension, use screen capture
      : await navigator.mediaDevices.getUserMedia({ video: { facingMode: "environment" } }); // Otherwise (app), use camera
    return stream;
  }

  return {
    init,
    stop,
    getLandmarks,
    getVideoSrc,
    handLandmarker,
    poseLandmarker,
    animationFrameId,
  };
}
