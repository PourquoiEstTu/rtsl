<script setup lang="ts">
import { ref } from 'vue'

// bind this with ML model output later
const translation = ref('Waiting for sign input...')
</script>

<template>
  <div
    class="flex flex-col w-72 h-full bg-[#FFF5B8]/90 text-black justify-center items-center p-4 border-l border-yellow-300"
  >
    <h2 class="text-lg font-bold mb-2">Translation:</h2>
    <p class="text-center text-base font-medium leading-relaxed">
      {{ translation }}
    </p>
  </div>
</template>
