<script setup lang="ts">
import { onMounted } from 'vue';

onMounted(async () => {
  const videoEl = document.getElementById('video') as HTMLMediaElement;

  let stream = null;
  try {
    stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "environment" } });
    videoEl.srcObject = stream;
  } catch (error) {
    console.error(error);
  }
})

</script>

<template>
  <video id="video" class="rounded-lg" autoplay playsinline />
</template>