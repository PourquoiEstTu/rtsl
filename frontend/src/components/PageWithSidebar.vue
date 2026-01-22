<script setup lang="ts">
import Sidebar from "./Sidebar.vue";
</script>

<template>
  <div class="page-layout">
    <Sidebar />
    <main class="page-main">
      <slot />
    </main>
  </div>
</template>

<style scoped>
/* Default: no sidebar layout (mobile/tablet) */
.page-layout {
  min-height: 100vh;
}

/* ✅ Laptop+: show sidebar + content */
@media (min-width: 1024px) {
  .page-layout {
    display: flex;
  }
  .page-main {
    flex: 1;
    min-width: 0;
  }
}
</style>
