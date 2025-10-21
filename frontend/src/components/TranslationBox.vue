<script setup lang="ts">
import { ref } from 'vue'

// bind this with ML model output later
const translation = ref('Waiting for sign input...')
</script>

<template>
    <div
        class="w-80 flex flex-col justify-center items-center
                bg-white/40 backdrop-blur-md rounded-r-[2.5rem] 
                px-6 text-gray-800 border-l border-white/30"
    >
        <h2 class="text-[#1E3A8A] font-bold text-lg mb-3"> Translation:</h2>
        <p class="text-gray-700 text-base italic"> {{ translation }}</p>
    </div>
</template>
