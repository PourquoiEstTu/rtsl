<script setup lang="ts">
import { onBeforeUnmount, onMounted, ref } from "vue";
import Button from "@/volt/Button.vue";
import TranslationBox from "@/components/TranslationBox.vue";
import { FilesetResolver, HandLandmarker, type HandLandmarkerResult } from "@mediapipe/tasks-vision";
import { useWebSocket } from "@vueuse/core";

let handLandmarker: HandLandmarker | null = null;
let animationFrameId: number | null = null;
let stream: MediaStream | null = null;

const videoEl = ref<HTMLVideoElement | null>(null);
const canvasEl = ref<HTMLCanvasElement | null>(null);

const { status, data, send, open, close } = useWebSocket("http://127.0.0.1:8000/ws");

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
    if (videoEl.value === null || canvasEl.value === null) {
      console.error("Missing references to video or canvas elements");
      return;
    }

    // Before we can use HandLandmarker class we must wait for it to finish
    // loading. Machine Learning models can be large and take a moment to
    // get everything needed to run.
    const createHandLandmarker = async () => {
      const vision = await FilesetResolver.forVisionTasks("/wasm");
      handLandmarker = await HandLandmarker.createFromOptions(vision, {
        baseOptions: {
          modelAssetPath: "/models/hand_landmarker.task",
          delegate: "GPU",
        },
        runningMode: "VIDEO",
        numHands: 2,
      });
    };

    await createHandLandmarker();

    if (handLandmarker === null) {
      console.error("Failed to initialize Hand Landmarker");
      return;
    }

    const canvasCtx = canvasEl.value.getContext("2d");
    if (canvasCtx === null) {
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
        if (!videoEl.value || !canvasEl.value) return;

        const videoWidth = videoEl.value.videoWidth;
        const videoHeight = videoEl.value.videoHeight;

        // Set canvas to video's intrinsic size
        canvasEl.value.width = videoWidth;
        canvasEl.value.height = videoHeight;

        const startTimeMs = performance.now();
        if (lastVideoTime !== videoEl.value.currentTime) {
          lastVideoTime = videoEl.value.currentTime;
          results = handLandmarker!.detectForVideo(videoEl.value, startTimeMs);
        }

        canvasCtx.clearRect(0, 0, canvasEl.value.width, canvasEl.value.height);

        if (results?.landmarks) {
          const W = canvasEl.value.width;
          const H = canvasEl.value.height;

          for (const landmarks of results.landmarks) {
            // Draw all 21 landmarks as circles
            for (const landmark of landmarks) {
              const x = landmark.x * W;
              const y = landmark.y * H;

              canvasCtx.beginPath();
              canvasCtx.arc(x, y, 5, 0, 2 * Math.PI);
              canvasCtx.fillStyle = "#00FF00";
              canvasCtx.fill();
            }

            // Draw connections between landmarks
            const connections = HandLandmarker.HAND_CONNECTIONS;
            canvasCtx.strokeStyle = "#00FF00";
            canvasCtx.lineWidth = 2;

            for (const conn of connections) {
              const start = landmarks[conn.start];
              const end = landmarks[conn.end];
              if (!start || !end) continue;

              canvasCtx.beginPath();
              canvasCtx.moveTo(start.x * W, start.y * H);
              canvasCtx.lineTo(end.x * W, end.y * H);
              canvasCtx.stroke();
            }
          }

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
</script>

<template>
  <div class="flex h-dvh w-dvw overflow-hidden bg-[#E0F2FF]">
    <aside
      class="hidden lg:flex w-52 bg-[#93c2e9]/90 flex-col items-center justify-center text-white font-semibold tracking-wide shadow-[4px_0_15px_rgba(0,0,0,0.1)] backdrop-blur-md rounded-r-3xl"
    >
      <span class="text-lg">MENU</span>
    </aside>
    <main class="flex-1 flex items-center justify-center p-6">
      <div
        class="relative w-full max-w-5xl h-[80vh] bg-white/20 backdrop-blur-md rounded-[2.5rem] border border-white/30 shadow-[0_8px_40px_rgba(0,0,0,0.1)] overflow-hidden flex"
      >
        <div class="relative flex-1 flex items-center justify-center rounded-l-[2.5rem] overflow-hidden">
          <video
            ref="videoEl"
            class="absolute top-0 left-0 w-full h-full object-cover bg-[#0F172A]"
            autoplay
            playsinline
          />
          <canvas ref="canvasEl" class="absolute w-full object-cover h-full" />
          <div class="absolute inset-0 bg-gradient-to-t from-black/40 via-transparent to-transparent"></div>

          <div class="absolute bottom-8 left-1/2 -translate-x-1/2 flex items-center justify-center">
            <Button
              class="!rounded-full !p-0 !m-0 !aspect-square w-14 sm:w-16 md:w-20 border-2 border-white/80 bg-white/10 backdrop-blur-md flex items-center justify-center hover:bg-white/20 active:scale-95 transition-all duration-200 shadow-[0_0_15px_rgba(255,255,255,0.3)]"
            >
              <i class="pi pi-play text-white text-2xl sm:text-3xl md:text-4xl drop-shadow-lg"></i>
            </Button>
          </div>
        </div>
        <TranslationBox />
      </div>
    </main>
  </div>
</template>
