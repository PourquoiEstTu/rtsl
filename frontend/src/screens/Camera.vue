<script setup lang="ts">
import { onMounted, onUnmounted, ref } from "vue";
import Button from "@/volt/Button.vue";
import TranslationBox from "@/components/TranslationBox.vue";
import Sidebar from "@/components/Sidebar.vue";

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
  <!-- Root layout -->
  <div class="flex h-dvh w-full overflow-hidden bg-[#E0F2FF]">

    <!-- Sidebar (visible on laptop only via Sidebar.vue CSS) -->
    <Sidebar />

    <!-- Main camera area -->
    <main class="flex-1 flex items-center justify-center p-6">
      <div
        class="relative w-full max-w-5xl h-[80vh]
               bg-white/20 backdrop-blur-md rounded-[2.5rem]
               border border-white/30 shadow-[0_8px_40px_rgba(0,0,0,0.1)]
               overflow-hidden flex"
      >

        <!-- Camera feed -->
        <div
          class="relative flex-1 flex items-center justify-center
                 rounded-l-[2.5rem] overflow-hidden"
        >
          <video
            ref="video"
            class="absolute top-0 left-0 w-full h-full object-cover bg-[#0F172A]"
            autoplay
            playsinline
          />

          <!-- Subtle overlay -->
          <div class="absolute inset-0 bg-gradient-to-t from-black/40 via-transparent to-transparent"></div>

          <!-- Play button -->
          <div
            class="absolute bottom-8 left-1/2 -translate-x-1/2
                   flex items-center justify-center"
          >
            <Button
              class="!rounded-full !p-0 !m-0 !aspect-square
                     w-14 sm:w-16 md:w-20
                     border-2 border-white/80
                     bg-white/10 backdrop-blur-md
                     flex items-center justify-center
                     hover:bg-white/20 active:scale-95
                     transition-all duration-200
                     shadow-[0_0_15px_rgba(255,255,255,0.3)]"
            >
              <i
                class="pi pi-play text-white text-2xl sm:text-3xl md:text-4xl drop-shadow-lg"
              ></i>
            </Button>
          </div>
        </div>

        <!-- Translation panel -->
        <TranslationBox />
      </div>
    </main>
  </div>
</template>
