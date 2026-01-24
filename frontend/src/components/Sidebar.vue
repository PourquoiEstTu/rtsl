<script setup lang="ts"></script>

<template>
  <nav class="sidebar__nav">
    <RouterLink class="sidebar__link" to="/">Home</RouterLink>
    <RouterLink class="sidebar__link" to="/camera">Translator</RouterLink>
    <RouterLink class="sidebar__link" to="/about">About</RouterLink>
  </nav>
</template>

<style scoped>
/* Hidden by default (phones/tablets) */
.sidebar {
  display: none;
}

/* Show on laptop+ */
@media (min-width: 1024px) {
  .sidebar__nav {
    display: flex;
    flex-direction: column;
    gap: 0.5rem;
  }

  .sidebar__link {
    padding: 0.6rem 0.75rem;
    border-radius: 10px;
    text-decoration: none;
    color: #1f2d3d;
    font-weight: 500;
  }

  /* Hover */
  .sidebar__link:hover {
    background: rgba(31, 45, 61, 0.1);
  }

  /* Active route */
  .sidebar__link.router-link-active {
    background: rgba(31, 45, 61, 0.2);
    font-weight: 700;
  }
}
</style>
