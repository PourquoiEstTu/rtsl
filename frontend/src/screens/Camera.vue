<script setup lang="ts">
import { computed, onMounted, onUnmounted, ref, watch } from "vue";
import Button from "@/volt/Button.vue";
import TranslationBox from "@/components/TranslationBox.vue";
import logo from "@/assets/logo_without_text-removebg-preview.png";
import "@/screens/style/camera.css";
import Sidebar from "@/components/Sidebar.vue";
import ChatHistoryButton from "@/components/ChatHistoryButton.vue";
import PhoneSidebarButton from "@/components/PhoneSidebarButton.vue";
import KeypointTransceiver from "@/components/KeypointTransceiver.vue";

// Track screen width
const screenWidth = ref(window.innerWidth);

// Update on resize
function updateWidth() {
  screenWidth.value = window.innerWidth;
}

// Define what counts as desktop/laptop
const isDesktop = computed(() => screenWidth.value >= 1024);

onMounted(() => {
  window.addEventListener("resize", updateWidth);
});
onUnmounted(() => {
  window.removeEventListener("resize", updateWidth);
});

const translatedWord = ref<string>("");
let wordTimeout: ReturnType<typeof setTimeout> | null = null;
const translatedSentence = ref<string>("Waiting for sign input...");
const translationHistory = ref<string[]>([]);

function onNewWord(word: string) {
  translatedWord.value = word;
  if (wordTimeout) clearTimeout(wordTimeout);
  wordTimeout = setTimeout(() => (translatedWord.value = ""), 1500);
}

function onNewSentence(sentence: string) {
  translatedSentence.value = sentence;
}

watch(translatedSentence, () => {
  translationHistory.value.push(translatedSentence.value);
});
</script>

<template>
  <div>
    <aside v-if="isDesktop" class="sidebar">
      <div>
        <span class="sidebar-title">Translator</span>
        <Sidebar />
      </div>
      <div class="logo">
        <img :src="logo" alt="RTSL Logo" class="camera-logo" />
        <h1 class="sidebar-title">RTSL</h1>
        <p class="sidebar-subtitle">Real-Time Sign Language</p>
      </div>
    </aside>
    <main>
      <div class="outer-container lg:p-2!">
        <div class="camera-panel h-full! lg:rounded-t-4xl">
          <KeypointTransceiver @new-word="onNewWord" @new-sentence="onNewSentence" />
          <div class="background-gradient"></div>

          <div class="relative w-full h-full">
            <div v-if="translatedWord" class="absolute inset-0 flex items-start justify-center pointer-events-none">
              <div class="px-4! py-2! mt-3! bg-gradient-to-b from-[#e9f6ff] to-[#a6d0f1] rounded-md text-black">
                {{ translatedWord }}
              </div>
            </div>
          </div>

          <ChatHistoryButton :translations="translationHistory" />
          <PhoneSidebarButton v-if="!isDesktop" />

          <div class="button-container">
            <Button class="button border-5! rounded-full!" />
          </div>
        </div>
        <TranslationBox class="translation-box" :translation="translatedSentence" />
      </div>
    </main>
  </div>
</template>
