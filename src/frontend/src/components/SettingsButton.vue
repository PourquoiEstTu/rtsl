<script setup lang="ts">
import { ref } from "vue";
import SettingsIcon from "./SettingsIcon.vue";
import Button from "@/volt/Button.vue";
import Dialog from "@/volt/Dialog.vue";
import { useLocalStorage } from "@vueuse/core";
import ToggleButton from "@/volt/ToggleButton.vue";

const visible = ref(false);

const drawKeypoints = useLocalStorage("drawKeypoints", true);
const tts = useLocalStorage("tts", true);
</script>

<template>
  <Button class="button rounded-md w-10!" @click="visible = true">
    <SettingsIcon />
  </Button>

  <Dialog v-model:visible="visible" header="Settings" class="w-9/10 lg:w-1/3 py-2! p-4!">
    <div class="flex justify-center items-center gap-4 p-4!">
      <span>Draw Keypoints</span>
      <ToggleButton pt:content:class="py-2! px-4!" v-model="drawKeypoints" on-label="On" off-label="Off" />
    </div>
    <div class="flex justify-center items-center gap-4 px-4! py-8!">
      <span>Text-to-Speech</span>
      <ToggleButton pt:content:class="py-2! px-4!" v-model="tts" on-label="On" off-label="Off" />
    </div>
  </Dialog>
</template>
