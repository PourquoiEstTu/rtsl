<script setup lang="ts">
import { ref, watch } from "vue";

const props = defineProps<{
  translation: string;
}>();

const display = ref("");
let jobId = 0;

function sleep(ms: number) {
  return new Promise((res) => setTimeout(res, ms));
}

watch(
  () => props.translation,
  async (newText) => {
    const id = ++jobId;

    // erase current text
    while (display.value.length > 0) {
      if (id !== jobId) return;
      display.value = display.value.slice(0, -1);
      await sleep(15);
    }

    // small pause (optional, improves UX)
    await sleep(100);

    // type new text
    for (let i = 0; i < newText.length; i++) {
      if (id !== jobId) return;
      display.value += newText[i];
      await sleep(25);
    }
  },
  { immediate: true },
);
</script>

<template>
  <div
    class="w-80 flex flex-col justify-center items-center bg-white backdrop-blur-md lg:rounded-b-4xl px-6! py-2! text-gray-800"
  >
    <h2 class="text-[#1E3A8A] font-bold text-lg mb-3">Translation:</h2>

    <p class="text-gray-700 text-base italic">{{ display }}</p>
  </div>
</template>
