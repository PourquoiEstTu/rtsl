<script setup lang="ts">
import ChatIcon from "@/components/ChatIcon.vue";
import Drawer from "@/volt/Drawer.vue";
import Button from "@/volt/Button.vue";
import { ref } from "vue";
import type { PropType } from "vue";

const props = defineProps({
  translations: {
    type: Array as PropType<string[]>,
    required: true,
  },
});

const visible = ref(false);
</script>

<template>
  <Drawer
    v-model:visible="visible"
    header="Chat History"
    position="right"
    pt:header:class="pl-4!"
    class="bg-gradient-to-b from-[#e9f6ff] to-[#a6d0f1] pb-4!"
  >
    <div class="flex flex-col! mt-4! gap-4! w-full h-full px-4!">
      <div v-if="translations.length" v-for="translation in translations" class="w-full flex justify-center">
        {{ translation }}
      </div>
      <div v-else>Translations from this session will appear here</div>
    </div>
  </Drawer>
  <Button class="button rounded-md w-10!" @click="visible = true">
    <ChatIcon />
  </Button>
</template>
