<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, ref, watch } from "vue";
import { useWebSocket } from "@vueuse/core";
import Toast from "@/volt/Toast.vue";
import useLandmarkerService from "@/composables/useLandmarkerService";
import { drawHandLandmarks, drawPoseLandmarks } from "@/utils/drawingUtils";
import { useToast } from "primevue/usetoast";

const props = defineProps({
  isOn: {
    type: Boolean,
    required: true,
  },
});

const isOnComputed = computed(() => props.isOn);

const landmarkerService = useLandmarkerService();
const toast = useToast();
const { status, data, send, open, close } = useWebSocket("wss://rtsl.cas.mcmaster.ca:8000/ws");

const emit = defineEmits(["newWord", "newSentence"]);

watch(data, (received: string) => {
  if (!received) return;
  const newData: { word: string; sentence: string } = JSON.parse(received);

  if (newData.word) {
    emit("newWord", newData.word);
  }
  if (newData.sentence) {
    emit("newSentence", newData.sentence);
  }
});

const videoEl = ref<HTMLVideoElement | null>(null);
const canvasEl = ref<HTMLCanvasElement | null>(null);

let lastSeenTime = performance.now();
let hasShownMissingToast = false;
const MISSING_THRESHOLD_MS = 3000;
const CLOSE_WS_THRESHOLD_MS = 15000;
const keepWebsocketClosed = ref(false);

watch(status, (newStatus) => {
  if (!keepWebsocketClosed.value && newStatus === "CLOSED") {
    console.log("Reopening WS connection.");
    open();
  }
});

onBeforeUnmount(landmarkerService.stop);

onMounted(async () => {
  if (!videoEl.value) {
    console.error("Missing references to video element");
    return;
  }
  if (!canvasEl.value) {
    console.error("Missing references to canvas element");
    return;
  }

  await landmarkerService.init(videoEl, false);

  const canvasCtx = canvasEl.value.getContext("2d");
  if (!canvasCtx) {
    console.error("Failed to get canvas context");
    return;
  }

  const predictWebcam = () => {
    if (!videoEl.value || !canvasEl.value) return;

    if (!isOnComputed.value) {
      canvasCtx.clearRect(0, 0, canvasEl.value.width, canvasEl.value.height);
      landmarkerService.animationFrameId.value = window.requestAnimationFrame(predictWebcam);
      return;
    }

    const videoWidth = videoEl.value.videoWidth;
    const videoHeight = videoEl.value.videoHeight;

    canvasEl.value.width = videoWidth;
    canvasEl.value.height = videoHeight;

    canvasCtx.clearRect(0, 0, canvasEl.value.width, canvasEl.value.height);

    const { handLandmarkerResults, poseLandmarkerResults } = landmarkerService.getLandmarks(videoEl, canvasEl);
    const hasLandmarks = !!handLandmarkerResults?.landmarks.length || !!poseLandmarkerResults?.landmarks.length;
    if (hasLandmarks) {
      lastSeenTime = performance.now();
      hasShownMissingToast = false;

      // Reopen WS if it was closed due to inactivity
      if (keepWebsocketClosed.value && status.value === "CLOSED") {
        keepWebsocketClosed.value = false;
        console.log("Reopening WS connection.");
        open();
      }

      send(
        JSON.stringify({
          hand: handLandmarkerResults,
          pose: poseLandmarkerResults,
        }),
      );
    }

    const now = performance.now();
    // Show toast if no landmarks for a while
    if (now - lastSeenTime > MISSING_THRESHOLD_MS && !hasShownMissingToast) {
      toast.add({
        severity: "warn",
        summary: "No detection",
        detail: "We can't see you. Make sure you're in frame.",
        life: 7000,
      });
      hasShownMissingToast = true;
    }
    // Close WS connection no landmarks in a long time
    if (now - lastSeenTime > CLOSE_WS_THRESHOLD_MS && status.value === "OPEN") {
      keepWebsocketClosed.value = true;
      console.log("Closing WS connection.");
      close();
    }

    if (handLandmarkerResults?.landmarks.length) {
      drawHandLandmarks(canvasCtx, handLandmarkerResults, canvasEl.value.width, canvasEl.value.height);
    }

    if (poseLandmarkerResults?.landmarks.length) {
      drawPoseLandmarks(canvasCtx, poseLandmarkerResults, canvasEl.value.width, canvasEl.value.height);
    }

    landmarkerService.animationFrameId.value = window.requestAnimationFrame(predictWebcam);
  };

  videoEl.value.addEventListener("loadeddata", predictWebcam);
});
</script>

<template>
  <Toast position="top-center" />
  <video ref="videoEl" autoplay playsinline />
  <canvas ref="canvasEl" class="absolute w-full object-cover h-full" />
</template>
