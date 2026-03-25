<script setup lang="ts">
import { ref } from "vue";
import KeypointTransceiver from "./KeypointTransceiver.vue";
import TranslationBox from "@/components/TranslationBox.vue";

const translatedSentence = ref<string>("Waiting for sign input...");

function onNewSentence(sentence: string) {
  translatedSentence.value = sentence;
}

const started = ref(false);

function startCapture() {
  started.value = true;
}

function stopCapture() {
  started.value = false;
}
</script>

<template>
  <div class="translator-shell">
    <div class="translator-card">
      <div class="left">
        <img src="/logo_cut.png" alt="RTSL logo" class="logo" />
        <div class="brand-text">
          <h1>RTSL</h1>
        </div>
      </div>

      <div class="center">
        <TranslationBox class="translation-box" :translation="translatedSentence" />
      </div>

      <div class="actions">
        <button v-if="!started" class="start-btn" @click="startCapture">Start Capture</button>

        <button v-else class="stop-btn" @click="stopCapture">Stop Capture</button>
      </div>

      <div v-if="started" class="hidden-capture">
        <KeypointTransceiver @new-sentence="onNewSentence" />
      </div>
    </div>
  </div>
</template>

<style scoped>
.translator-shell {
  width: 100%;
  height: 100vh;
  margin: 0;
  padding: 8px;
  box-sizing: border-box;
  background: linear-gradient(180deg, #d9ecfb 0%, #c9e0f5 100%);
  font-family:
    Inter,
    ui-sans-serif,
    system-ui,
    -apple-system,
    BlinkMacSystemFont,
    "Segoe UI",
    sans-serif;
}

.translator-card {
  height: calc(100vh - 16px);
  border-radius: 20px;
  background: rgba(255, 255, 255, 0.72);
  border: 1px solid rgba(255, 255, 255, 0.7);
  box-shadow: 0 10px 24px rgba(44, 88, 129, 0.12);
  padding: 10px 16px;
  display: flex;
  align-items: center;
  gap: 16px;
  box-sizing: border-box;
}

.left {
  display: flex;
  align-items: center;
  gap: 12px;
  min-width: 220px;
}

.logo {
  width: 42px;
  height: 42px;
  object-fit: contain;
  flex-shrink: 0;
}

.brand-text h1 {
  margin: 0;
  font-size: 1.1rem;
  line-height: 1;
  color: #1c2b4a;
  font-weight: 800;
}

.brand-text p {
  margin: 4px 0 0;
  font-size: 0.72rem;
  color: #52627e;
}

.center {
  flex: 1;
  display: flex;
  align-items: center;
}

.translation-box {
  flex: 0.5;
  width: 100%; /* ensure full width */
}

.hidden-capture {
  position: fixed;
  width: 1px;
  height: 1px;
  overflow: hidden;
  opacity: 0;
  pointer-events: none;
}

.start-btn,
.stop-btn {
  border: none;
  border-radius: 14px;
  padding: 10px 14px;
  font-size: 0.85rem;
  font-weight: 700;
  cursor: pointer;
  flex-shrink: 0;
}

.start-btn {
  background: #274690;
  color: white;
}

.stop-btn {
  background: #dc2626;
  color: white;
}
</style>
