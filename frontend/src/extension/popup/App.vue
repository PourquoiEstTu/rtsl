<script setup lang="ts">
import { ref } from "vue";

const status = ref("Ready");
const logo = "/logo_cut.png";
async function openTranslator() {
  try {
    await chrome.windows.create({
      url: chrome.runtime.getURL("src/extension/translator.html"),
      type: "popup",
      width: 900,
      height: 400,
      focused: true
    });
    status.value = "Translator opened.";
  } catch (error) {
    console.error(error);
    status.value = "Failed to open translator.";
  }
}
</script>

<template>
  <div class="popup-shell">
    <div class="popup-card">
      <img :src="logo" alt="RTSL logo" class="logo" />

      <h1>RTSL</h1>
      <p class="subtitle">Real-Time Sign Language Assistant</p>

      <button class="primary-btn" @click="openTranslator">
        Open Translator
      </button>

      <div class="helper-text">
        Select your meeting tab when the share prompt appears.
      </div>

      <!-- <div class="status-pill">
        {{ status }}
      </div> -->
    </div>
  </div>
</template>

<style scoped>
.popup-shell {
  width: 340px;
  min-height: 420px;
  padding: 18px;
  background: linear-gradient(180deg, #d9ecfb 0%, #c9e0f5 100%);
  font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont,
    "Segoe UI", sans-serif;
  color: #1f2a44;
  box-sizing: border-box;
}

.popup-card {
  height: 100%;
  background: rgba(255, 255, 255, 0.72);
  border: 1px solid rgba(255, 255, 255, 0.7);
  border-radius: 24px;
  padding: 24px 20px;
  box-shadow: 0 18px 40px rgba(44, 88, 129, 0.16);
  display: flex;
  flex-direction: column;
  align-items: center;
  text-align: center;
}

.logo {
  width: 76px;
  height: 76px;
  object-fit: contain;
  margin-bottom: 10px;
}

h1 {
  margin: 0;
  font-size: 2rem;
  font-weight: 800;
  letter-spacing: -0.03em;
  color: #1c2b4a;
}

.subtitle {
  margin: 8px 0 24px;
  font-size: 0.98rem;
  line-height: 1.4;
  color: #4e5d78;
}

.primary-btn {
  width: 100%;
  border: none;
  border-radius: 18px;
  padding: 16px 18px;
  font-size: 1.05rem;
  font-weight: 700;
  cursor: pointer;
  background: #ffffff;
  color: #274690;
  box-shadow: 0 10px 24px rgba(75, 120, 170, 0.16);
  transition: transform 0.15s ease, box-shadow 0.15s ease;
}

.primary-btn:hover {
  transform: translateY(-1px);
  box-shadow: 0 14px 28px rgba(75, 120, 170, 0.2);
}

.primary-btn:active {
  transform: translateY(0);
}

.helper-text {
  margin-top: 16px;
  font-size: 0.9rem;
  line-height: 1.45;
  color: #586782;
}

.status-pill {
  margin-top: 20px;
  padding: 10px 14px;
  border-radius: 999px;
  background: rgba(39, 70, 144, 0.1);
  color: #274690;
  font-size: 0.88rem;
  font-weight: 600;
}
</style>