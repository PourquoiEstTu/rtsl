<template>
  <Toast
    unstyled
    :pt="theme"
    :ptOptions="{
      mergeProps: ptViewMerge,
    }"
  >
    <template #closeicon>
      <TimesIcon />
    </template>
    <template v-for="(_, slotName) in $slots" #[slotName]="slotProps">
      <slot :name="slotName" v-bind="slotProps ?? {}" />
    </template>
  </Toast>
</template>

<script setup lang="ts">
import TimesIcon from "@primevue/icons/times";
import Toast, { type ToastPassThroughOptions, type ToastProps } from "primevue/toast";
import { ref } from "vue";
import { ptViewMerge } from "./utils";

interface Props extends /* @vue-ignore */ ToastProps {}
defineProps<Props>();

const theme = ref<ToastPassThroughOptions>({
  root: `w-96 rounded-md whitespace-pre-line break-words
        p-top-center:-translate-x-1/2 p-bottom-center:-translate-x-1/2
        p-center:min-w-[20vw] p-center:-translate-x-1/2 p-center:-translate-y-1/2`,
  message: `mb-4 px-4! py-2! not-p-custom:border not-p-custom:backdrop-blur-sm  not-p-custom:rounded-md
        p-info:bg-blue-50/95 p-info:border-blue-200 p-info:text-blue-600   
        p-success:bg-green-50/95 p-success:border-green-200 p-success:text-green-600   
        p-warn:bg-yellow-50/95 p-warn:border-yellow-200 p-warn:text-yellow-600   
        p-error:bg-red-50/95 p-error:border-red-200 p-error:text-red-600   
        p-secondary:bg-surface-100 p-secondary:border-surface-200 p-secondary:text-surface-600   
        p-contrast:bg-surface-900 p-contrast:border-surface-950 p-contrast:text-surface-50 `,
  messageContent: `flex items-start p-3 gap-2`,
  messageIcon: `flex-shrink-0 text-lg w-[1.125rem] h-[1.125rem] mt-1`,
  messageText: `flex-auto flex flex-col gap-2`,
  summary: `font-medium text-base`,
  detail: `font-medium text-sm text-surface-700
        p-contrast:text-surface-0`,
  buttonContainer: ``,
  closeButton: `flex items-center justify-center overflow-hidden relative cursor-pointer bg-transparent select-none
        transition-colors duration-200 text-inherit w-7 h-7 rounded-full -mt-[25%] -end-1/4 p-0 border-none
        focus-visible:outline focus-visible:outline-1 focus-visible:outline-offset-2
        p-info:hover:bg-blue-100 p-info:focus-visible:outline-blue-600  
        p-success:hover:bg-green-100 p-success:focus-visible:outline-green-600  
        p-warn:hover:bg-yellow-100 p-warn:focus-visible:outline-yellow-600  
        p-error:hover:bg-red-100 p-error:focus-visible:outline-red-600  
        p-secondary:hover:bg-surface-200 p-secondary:focus-visible:outline-surface-600  
        p-contrast:hover:bg-surface-800 p-contrast:focus-visible:outline-surface-50 `,
  closeIcon: `text-base w-4 h-4`,
  transition: {
    enterFromClass: "opacity-0 translate-y-1/2",
    enterActiveClass: "transition-all duration-500",
    leaveFromClass: "max-h-[1000px]",
    leaveActiveClass: "transition-all duration-500",
    leaveToClass: "max-h-0 opacity-0 mb-0 overflow-hidden",
  },
});
</script>
