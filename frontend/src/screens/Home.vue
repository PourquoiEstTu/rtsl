<script setup lang="ts">
import logo from '@/assets/logo_without_text-removebg-preview.png';
import { useRouter } from 'vue-router';
import "@/screens/style/home.css";

const router = useRouter()

function goToCamera() {
  console.log("Navigating to camera...");
  router.push('/camera')
}

function goToWebExtension() {
  console.log("Navigating to web extension...");
  router.push('/web_extension')
}

function goToAbout() {
  console.log("Navigating to about...");
  router.push('/about')
}
</script>

<template>
  <div class="home-wrapper">
    <!-- Header -->
    <div class="home-header">
      <img :src="logo" alt="RTSL Logo" class="home-logo" />

      <h1 class="home-title">RTSL</h1>
      <p class="home-subtitle">Real-Time Sign Language Assistant</p>
    </div>

    <!-- Cards Section -->
    <div class="home-card-row">

      <div class="home-card" @click="goToCamera">
        <div class="home-icon">
          <i class="pi pi-camera"></i>
        </div>
        <h2 class="home-card-title">
          Open Camera<br />Translator
        </h2>
        <p class="home-card-desc">Translate ASL to English</p>
      </div>

      <div class="home-card" @click="goToWebExtension">
        <div class="home-icon">
          <i class="pi pi-microchip"></i>
        </div>
        <h2 class="home-card-title">
          Download the<br />Web Extension
        </h2>
        <p class="home-card-desc">Zoom / Meet / Teams</p>
      </div>

      <div class="home-card" @click="goToAbout">
        <div class="home-icon">
          <i class="pi pi-question"></i>
        </div>
        <h2 class="home-card-title">
          About &<br />How to Use
        </h2>
        <p class="home-card-desc">Setup-guide + Tips</p>
      </div>
    </div>
  </div>
</template>
