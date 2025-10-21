<script setup lang="ts">
import { onMounted } from 'vue';
import Button from '@/volt/Button.vue';
import EyeIcon from '@primevue/icons/eye';
import TranslationBox from '@/components/TranslationBox.vue'

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
  <div class="flex h-dvh w-dvw overflow-hidden bg-black">
    <div
      class="hidden md:flex w-48 bg-[#FFFC00]/80 items-center justify-center text-black font-semibold"
    >
      MENU
    </div>
    <div class="relative flex-1 flex items-center justify-center">
      <video
        id="video"
        class="absolute top-0 left-0 w-full h-full object-cover bg-black"
        autoplay
        playsinline
      />
      <Button
        :class="`
          px-3! absolute! bottom-12 left-1/2 -translate-x-1/2 
          border-4 border-white backdrop-blur-lg bg-transparent rounded-full! w-24 h-24
          active:scale-95 transition
        `"
        size="large"
      >
        <EyeIcon class="w-16 h-16 text-white" />
      </Button>
    </div>
    <TranslationBox />
  </div>
</template>