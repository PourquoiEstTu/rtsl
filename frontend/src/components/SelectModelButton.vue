<script setup lang="ts">
import Button from "@/volt/Button.vue";
import ModelIcon from "@/components/ModelIcon.vue";
import Dialog from "@/volt/Dialog.vue";
import Listbox from "@/volt/Listbox.vue";
import { ref } from "vue";
import { useLocalStorage } from "@vueuse/core";

const visible = ref(false);

const MODEL_OPTIONS = [
  "Showcase",
  "Self-Trained 100",
  "Self-Trained 300",
  "Self-Trained 1000",
  "Self-Trained 2000",
  "Pre-Trained 100",
  "Pre-Trained 300",
  "Pre-Trained 1000",
  "Pre-Trained 2000",
] as const;
type ModelOption = (typeof MODEL_OPTIONS)[number];

const selectedModel = useLocalStorage<ModelOption>("model", "Showcase");
const modelOptions = ref<ModelOption[]>([...MODEL_OPTIONS]);

function onChange(model: ModelOption | null) {
  if (model === null || selectedModel.value === model) return;
  selectedModel.value = model;
}
</script>

<template>
  <Button class="button rounded-md w-10!" @click="visible = true">
    <ModelIcon />
  </Button>

  <Dialog v-model:visible="visible" header="Select Model" class="w-9/10 lg:w-1/3 py-2! p-4!">
    <div class="flex justify-center items-center">
      <Listbox
        pt:listContainer:class="max-h-none!"
        pt:option:class="py-2! px-4! select-none!"
        :model-value="selectedModel"
        @update:modelValue="onChange"
        :options="modelOptions"
      />
    </div>
  </Dialog>
</template>
