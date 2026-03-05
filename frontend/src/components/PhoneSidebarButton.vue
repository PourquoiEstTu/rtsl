<script setup lang="ts">
import Drawer from '@/volt/Drawer.vue';
import PhoneSidebarIcon from './PhoneSidebarIcon.vue';
import Sidebar from './Sidebar.vue';
import { ref } from 'vue';
import logo from "@/assets/logo_without_text-removebg-preview.png";

const visible = ref(false);

</script>

<template>
  <Drawer v-model:visible="visible" header="Menu" pt:header:class="pl-4!"
    class="bg-gradient-to-b from-[#e9f6ff] to-[#a6d0f1]">
    <div class="flex flex-col h-full justify-between px-2! pb-8! rounded-0!">
      <Sidebar />
      <div class="logo">
        <img :src="logo" alt="RTSL Logo" class="camera-logo" />
        <h1 class="sidebar-title">RTSL</h1>
        <p class="sidebar-subtitle">Real-Time Sign Language</p>
      </div>
    </div>
  </Drawer>
  <Button class="button rounded-md w-10! absolute! top-[2%] left-[5%]" @click="visible = true">
    <PhoneSidebarIcon />
  </Button>
</template>