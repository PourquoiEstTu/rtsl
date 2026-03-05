<script setup lang="ts">

// bind this with ML model output later
const props = defineProps({
    translation: {
        type: String,
        required: true
    }
});

</script>
<template>
    <div class="w-80 flex flex-col justify-center items-center
                bg-white/40 backdrop-blur-md rounded-b-4xl 
                px-6 text-gray-800">
        <h2 class="text-[#1E3A8A] font-bold text-lg mb-3"> Translation:</h2>
        <p class="text-gray-700 text-base italic"> {{ translation }}</p>
    </div>
</template>