<script setup lang="ts">
import { onMounted, onUnmounted, ref } from "vue";
import Button from "@/volt/Button.vue";
import TranslationBox from "@/components/TranslationBox.vue";
import "@/screens/style/camera.css";

const video = ref<HTMLVideoElement | null>(null);
let stream: MediaStream | null = null;

onMounted(async () => {
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      video: true,
      audio: false,
    });
    if (video.value) {
      video.value.srcObject = stream;
    }
  } catch (err) {
    console.error("Camera error:", err);
  }
});

onUnmounted(() => {
  if (stream) {
    stream.getTracks().forEach((track) => track.stop());
    console.log("Camera stopped");
  }
});
</script>

<template>
  <div class="camera-wrapper">
    <aside class="sidebar">
      <span class="text-lg">MENU</span>
    </aside>
    <main>
      <div class="outer-container">
        <div class="camera-panel">
          <video ref="video" autoplay playsinline />
          <div class="background-gradient"></div>

          <div class="button-container">
            <Button class="button">
              <i class="play-icon pi pi-play"></i>
            </Button>
          </div>
        </div>
        <TranslationBox class="translation-box" />
      </div>
    </main>
  </div>
</template>
