<script setup lang="ts">
import KeypointTransceiver from "@/components/KeypointTransceiver.vue";
</script>

<template>
  <div class="translator-shell">
    <div class="header">
      <h1>RTSL Translator</h1>
      <p>Select your meeting tab when the screen-share prompt appears.</p>
    </div>

    <div class="content">
      <KeypointTransceiver />
    </div>
  </div>
</template>

<style scoped>
.translator-shell {
  min-height: 100vh;
  background: #020617;
  color: white;
  padding: 20px;
  font-family: Arial, sans-serif;
}

.header {
  margin-bottom: 16px;
}

.header h1 {
  margin: 0 0 8px;
  font-size: 24px;
}

.header p {
  margin: 0;
  color: #94a3b8;
}

.content {
  position: relative;
  min-height: 70vh;
  background: #0f172a;
  border: 1px solid #1e293b;
  border-radius: 16px;
  overflow: hidden;
}
</style>
