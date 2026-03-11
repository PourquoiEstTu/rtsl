<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, onUnmounted, ref, watch } from "vue";
import Button from "@/volt/Button.vue";
import TranslationBox from "@/components/TranslationBox.vue";
import {
  FilesetResolver,
  HandLandmarker,
  type HandLandmarkerResult,
} from "@mediapipe/tasks-vision";
import { useWebSocket } from "@vueuse/core";
import logo from "@/assets/logo_without_text-removebg-preview.png";
import "@/screens/style/camera.css";
import Sidebar from "@/components/Sidebar.vue";
import ChatHistoryButton from "@/components/ChatHistoryButton.vue";
import PhoneSidebarButton from "@/components/PhoneSidebarButton.vue";

// Track screen width
const screenWidth = ref(window.innerWidth);

// Update on resize
function updateWidth() {
  screenWidth.value = window.innerWidth;
}

onMounted(() => {
  window.addEventListener("resize", updateWidth);
});
onUnmounted(() => {
  window.removeEventListener("resize", updateWidth);
});
// Define what counts as desktop/laptop
const isDesktop = computed(() => screenWidth.value >= 1024);

let handLandmarker: HandLandmarker | null = null;
let animationFrameId: number | null = null;
let stream: MediaStream | null = null;

const videoEl = ref<HTMLVideoElement | null>(null);
const canvasEl = ref<HTMLCanvasElement | null>(null);

// Todo: assign string returned from backend to this var
const lastTranslation = ref<string>("Waiting for sign input...");
// These are temparary for demo
const exmapleTranslations = [
  'Idk brochacho ✌️😭',
  'Long long long long long long long long long long long long long long long long long long long long long long long long long long long long long long long long sentence from a yapper',
  'Test 1',
  'Test 2',
  'Test 3',
  'Test 4',
  'Test 5',
  'Test 6',
  'Test 7',
  'Lol 6 7',
];

const translationHistory = ref<string[]>([]);

watch(lastTranslation, () => {
  translationHistory.value.push(lastTranslation.value)
})

function* exampleTranslation() {
  while (true) {
    for (const translation of exmapleTranslations) {
      yield translation;
    }
  }
}

const genExampleTranslation = exampleTranslation();

// Todo for this coding sess:
// * "convo history" button to show all translations from this session

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

    // Request camera access
    stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: "environment" },
    });
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
  const vision = await FilesetResolver.forVisionTasks("/wasm");
  return await HandLandmarker.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath: "/models/hand_landmarker.task",
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
  <div>
    <aside v-if="isDesktop" class="sidebar">
      <div>
        <span class="sidebar-title">Translator</span>
        <Sidebar />
      </div>
      <div class="logo">
        <img :src="logo" alt="RTSL Logo" class="camera-logo" />
        <h1 class="sidebar-title">RTSL</h1>
        <p class="sidebar-subtitle">Real-Time Sign Language</p>
      </div>
    </aside>
    <main>
      <div class="outer-container lg:p-2!">
        <div class="camera-panel lg:rounded-t-4xl">
          <video ref="videoEl" autoplay playsinline />
          <canvas ref="canvasEl" class="absolute w-full object-cover h-full" />
          <div class="background-gradient"></div>
          <ChatHistoryButton :translations="translationHistory" />
          <PhoneSidebarButton v-if="!isDesktop" />

          <div class="button-container">
            <Button class="button border-5! rounded-full!"
              @click="lastTranslation = genExampleTranslation.next().value as string" />
          </div>
        </div>
        <TranslationBox class="translation-box" :translation="lastTranslation" />
      </div>
    </main>
  </div>
</template>
