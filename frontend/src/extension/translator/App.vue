<script setup lang="ts">
import { onBeforeUnmount, onMounted, ref } from "vue";
import { useTranslatorSession } from "../useTranslatorSession";

const videoEl = ref<HTMLVideoElement | null>(null);
const canvasEl = ref<HTMLCanvasElement | null>(null);

const {
  sessionStatus,
  socketStatus,
  translationText,
  debugText,
  sourceTitle,
  setDebug,
  initMediaPipe,
  consumeCapturedTabStream,
  startLoop,
  stopCapture,
  cleanup
} = useTranslatorSession();

function startIfReady() {
  if (!videoEl.value || !canvasEl.value) return;
  startLoop(videoEl.value, canvasEl.value);
}

chrome.runtime.onMessage.addListener((message) => {
  if (message.type === "START_CAPTURE_WITH_STREAM_ID") {
    if (!videoEl.value || !canvasEl.value) return;

    consumeCapturedTabStream(
      message.streamId,
      message.tabTitle ?? "Meeting Tab",
      videoEl.value
    )
      .then(() => {
        startLoop(videoEl.value!, canvasEl.value!);
      })
      .catch((error) => {
        setDebug(
          error instanceof Error ? error.stack ?? error.message : String(error)
        );
      });
  }

  if (message.type === "STOP_CAPTURE") {
    stopCapture();
  }
});

onMounted(async () => {
  try {
    await initMediaPipe();
    setDebug("MediaPipe initialized. Waiting for stream ID...");
    await chrome.runtime.sendMessage({ type: "TRANSLATOR_READY" });
  } catch (error) {
    setDebug(
      error instanceof Error ? error.stack ?? error.message : String(error)
    );
  }
});

onBeforeUnmount(() => {
  cleanup();
});
</script>

<template>
  <div class="app-shell">
    <aside class="sidebar">
      <h1>RTSL</h1>
      <p>Meeting Tab Translator</p>

      <div class="meta">
        <div><strong>Status:</strong> {{ sessionStatus }}</div>
        <div><strong>Source:</strong> {{ sourceTitle }}</div>
        <div><strong>Socket:</strong> {{ socketStatus }}</div>
      </div>

      <div class="controls">
        <button @click="startIfReady">Start Loop</button>
        <button @click="stopCapture">Stop</button>
      </div>
    </aside>

    <main class="main-panel">
      <section class="video-panel">
        <video ref="videoEl" autoplay playsinline muted></video>
        <canvas ref="canvasEl"></canvas>
      </section>

      <section class="translation-panel">
        <h2>Live Translation</h2>
        <div class="translation-text">{{ translationText }}</div>
        <pre class="debug-text">{{ debugText }}</pre>
      </section>
    </main>
  </div>
</template>

<style scoped>
* {
  box-sizing: border-box;
}

.app-shell {
  display: grid;
  grid-template-columns: 280px 1fr;
  min-height: 100vh;
  font-family: Arial, sans-serif;
  background: #020617;
  color: #e2e8f0;
}

.sidebar {
  background: #0f172a;
  border-right: 1px solid #1e293b;
  padding: 20px;
}

.sidebar h1 {
  margin: 0 0 8px;
}

.sidebar p {
  margin: 0 0 20px;
  color: #94a3b8;
}

.meta {
  display: grid;
  gap: 10px;
  font-size: 14px;
  margin-bottom: 20px;
}

.controls {
  display: grid;
  gap: 10px;
}

.controls button {
  padding: 12px;
  border: none;
  border-radius: 10px;
  cursor: pointer;
  background: #334155;
  color: white;
}

.main-panel {
  display: grid;
  grid-template-columns: 1.2fr 0.8fr;
  gap: 20px;
  padding: 20px;
}

.video-panel,
.translation-panel {
  background: #0f172a;
  border: 1px solid #1e293b;
  border-radius: 18px;
  padding: 16px;
}

.video-panel {
  position: relative;
  min-height: 520px;
}

video,
canvas {
  width: 100%;
  max-height: 75vh;
  border-radius: 14px;
  background: black;
}

canvas {
  position: absolute;
  top: 16px;
  left: 16px;
  width: calc(100% - 32px);
  height: auto;
  pointer-events: none;
}

.translation-panel h2 {
  margin-top: 0;
}

.translation-text {
  min-height: 140px;
  background: #020617;
  border-radius: 12px;
  padding: 16px;
  font-size: 20px;
  line-height: 1.5;
  white-space: pre-wrap;
}

.debug-text {
  margin-top: 16px;
  background: #020617;
  border-radius: 12px;
  padding: 12px;
  max-height: 260px;
  overflow: auto;
  font-size: 12px;
  color: #94a3b8;
}
</style>