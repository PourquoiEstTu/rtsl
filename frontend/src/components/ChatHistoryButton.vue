<script setup lang="ts">
import ChatIcon from "@/components/ChatIcon.vue";
import Drawer from "@/volt/Drawer.vue";
import { ref } from "vue";
import type { PropType } from "vue";

const props = defineProps({
  translations: {
    type: Array as PropType<string[]>,
    required: true,
  }
})

const visible = ref(false);

</script>

<template>
  <Drawer v-model:visible="visible" header="Chat History" position="right" pt:header:class="pl-4!"
    class="bg-gradient-to-b from-[#e9f6ff] to-[#a6d0f1]">
    <div class="flex flex-col! mt-4! gap-4! w-full h-full px-4! pb-2!">
      <div v-for="translation in translations" class="w-full flex justify-center">
        {{ translation }}
      </div>
    </div>
  </Drawer>
  <Button class="button rounded-md w-10! absolute! top-[2%] right-[5%] lg:right-[2%]" @click="visible = true">
    <ChatIcon />
  </Button>
</template>