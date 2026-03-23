<script setup lang="ts">
import { computed } from "vue";

const props = defineProps<{
  translation: string;
}>();

const charCount = computed(() => props.translation.length);

// Calculate duration: e.g., 0.05 seconds per character
const typingDuration = computed(() => `${charCount.value * 0.05}s`);
</script>

<template>
  <div
    class="w-80 flex flex-col justify-center items-center bg-white backdrop-blur-md lg:rounded-b-4xl px-6 py-2 text-gray-800"
  >
    <h2 class="text-[#1E3A8A] font-bold text-lg mb-3">Translation:</h2>

    <p
      :key="translation"
      class="typing-effect font-mono text-gray-700 text-base overflow-hidden whitespace-nowrap border-r-2 border-[#1E3A8A]"
      :style="{
        '--chars': charCount,
        '--duration': typingDuration,
      }"
    >
      {{ translation }}
    </p>
  </div>
</template>

<style scoped>
.typing-effect {
  width: 0;
  /* Use the dynamic --duration variable here */
  animation:
    typing var(--duration) steps(var(--chars)) forwards,
    blink 0.8s step-end infinite;
}

@keyframes typing {
  from {
    width: 0;
  }
  to {
    width: calc(var(--chars) * 1ch);
  }
}

@keyframes blink {
  from,
  to {
    border-color: transparent;
  }
  50% {
    border-color: #1e3a8a;
  }
}
</style>
