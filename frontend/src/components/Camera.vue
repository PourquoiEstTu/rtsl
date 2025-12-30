<script setup lang="ts">
import { onBeforeUnmount, onMounted } from "vue";
import Button from "@/volt/Button.vue";
import TranslationBox from "@/components/TranslationBox.vue";
import { FilesetResolver, HandLandmarker, type HandLandmarkerResult } from "@mediapipe/tasks-vision";
import { useWebSocket } from "@vueuse/core";

let handLandmarker: HandLandmarker | null = null;
let animationFrameId: number | null = null;

const { status, data, send, open, close } = useWebSocket("http://127.0.0.1:8000/ws");

onBeforeUnmount(() => {
  handLandmarker?.close();

  // Add cleanup for video stream
  const videoEl = document.getElementById("webcam") as HTMLVideoElement;
  if (videoEl?.srcObject) {
    const stream = videoEl.srcObject as MediaStream;
    stream.getTracks().forEach((track) => track.stop());
    videoEl.srcObject = null;
  }

  // Cancel animation frame
  if (animationFrameId) {
    cancelAnimationFrame(animationFrameId);
  }
});

onMounted(async () => {
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
    console.error("Failed to initialize Hand Land Marker");
    return;
  }

  const videoEl = document.getElementById("webcam") as HTMLVideoElement;
  const canvasEl = document.getElementById("output-canvas") as HTMLCanvasElement;
  const canvasCtx = canvasEl.getContext("2d");
  if (canvasCtx === null) {
    console.error("Failed to get canvas context");
    return;
  }

  const stream = await navigator.mediaDevices.getUserMedia({
    video: { facingMode: "environment" },
  });
  videoEl.srcObject = stream;
  videoEl.addEventListener("loadeddata", predictWebcam);

  let lastVideoTime = -1;
  let results: HandLandmarkerResult | undefined = undefined;

  async function predictWebcam() {
    const videoWidth = videoEl.videoWidth;
    const videoHeight = videoEl.videoHeight;

    // Set canvas to video's intrinsic size
    canvasEl.width = videoWidth;
    canvasEl.height = videoHeight;

    let startTimeMs = performance.now();
    if (lastVideoTime !== videoEl.currentTime) {
      lastVideoTime = videoEl.currentTime;
      results = handLandmarker!.detectForVideo(videoEl, startTimeMs);
    }

    canvasCtx!.save();
    canvasCtx!.clearRect(0, 0, canvasEl.width, canvasEl.height);
    if (results?.landmarks) {
      // Use the video's intrinsic dimensions for de-normalization,
      // as per the last successful attempt configuration.
      const W = canvasEl.width;
      const H = canvasEl.height;

      for (const landmarks of results.landmarks) {
        // Draw all 21 landmarks as circles
        for (const landmark of landmarks) {
          // De-normalize the coordinates (0-1) to pixel values (0-W/H)
          const x = landmark.x * W;
          const y = landmark.y * H;

          canvasCtx!.beginPath();
          canvasCtx!.arc(x, y, 5, 0, 2 * Math.PI); // Draw a circle with radius 5
          canvasCtx!.fillStyle = "#00FF00"; // Bright Green
          canvasCtx!.fill();
        }

        for (const landmarks of results.landmarks) {
          const connections = HandLandmarker.HAND_CONNECTIONS;

          canvasCtx!.strokeStyle = "#00FF00";
          canvasCtx!.lineWidth = 2;

          for (const conn of connections) {
            const start = landmarks[conn.start];
            const end = landmarks[conn.end];
            if (!start || !end) continue;

            canvasCtx!.beginPath();
            canvasCtx!.moveTo(start.x * W, start.y * H);
            canvasCtx!.lineTo(end.x * W, end.y * H);
            canvasCtx!.stroke();
          }
        }
      }

      send(JSON.stringify(results));
    }
    canvasCtx!.restore();
    animationFrameId = window.requestAnimationFrame(predictWebcam);
  }
});
</script>

<template>
  <div class="flex h-dvh w-dvw overflow-hidden bg-[#E0F2FF]">
    <main class="flex-1 flex items-center justify-center p-6">
      <div
        class="relative w-full max-w-5xl h-[80vh] bg-white/20 backdrop-blur-md rounded-[2.5rem] border border-white/30 shadow-[0_8px_40px_rgba(0,0,0,0.1)] overflow-hidden flex"
      >
        <div class="relative flex-1 flex items-center justify-center rounded-l-[2.5rem] overflow-hidden">
          <!-- Both video and canvas MUST have same size, which is why object-cover is applied to both -->
          <video id="webcam" class="absolute w-full h-full object-cover bg-[#0F172A] p-0!" playsinline autoplay />
          <canvas id="output-canvas" class="absolute w-full object-cover h-full" />
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
