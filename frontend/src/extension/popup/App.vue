<script setup lang="ts">
import { ref } from "vue";

const status = ref("Idle");

async function startTranslator() {
  try {
    status.value = "Looking for active tab...";

    const [tab] = await chrome.tabs.query({
      active: true,
      lastFocusedWindow: true
    });

    if (!tab?.id) {
      status.value = "No active tab found.";
      return;
    }

    await chrome.runtime.sendMessage({
      type: "START_TRANSLATOR",
      targetTabId: tab.id,
      tabTitle: tab.title
    });

    status.value = "Translator started.";
  } catch (error) {
    console.error(error);
    status.value =
      error instanceof Error ? error.message : "Failed to start translator.";
  }
}

async function stopTranslator() {
  try {
    await chrome.runtime.sendMessage({
      type: "STOP_TRANSLATOR"
    });
    status.value = "Stop signal sent.";
  } catch (error) {
    console.error(error);
    status.value =
      error instanceof Error ? error.message : "Failed to stop translator.";
  }
}
</script>

<template>
  <div class="popup-shell">
    <h1>RTSL Translator</h1>
    <p class="desc">
      Open your meeting tab first, then click Start Translator.
    </p>

    <button class="start" @click="startTranslator">Start Translator</button>
    <button class="stop" @click="stopTranslator">Stop Translator</button>

    <div class="status">{{ status }}</div>
  </div>
</template>

<style scoped>
.popup-shell {
  width: 320px;
  padding: 16px;
  font-family: Arial, sans-serif;
  background: #0f172a;
  color: white;
}

h1 {
  margin: 0 0 8px;
  font-size: 18px;
}

.desc {
  margin: 0 0 16px;
  color: #cbd5e1;
  font-size: 13px;
  line-height: 1.5;
}

button {
  width: 100%;
  padding: 12px;
  border: none;
  border-radius: 10px;
  cursor: pointer;
  margin-bottom: 10px;
  font-size: 14px;
  font-weight: 600;
}

.start {
  background: #22c55e;
  color: #052e16;
}

.stop {
  background: #ef4444;
  color: white;
}

.status {
  margin-top: 10px;
  font-size: 12px;
  color: #93c5fd;
  white-space: pre-wrap;
}
</style>