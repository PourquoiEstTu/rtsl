<script setup lang="ts"></script>

<template>
  <aside class="sidebar">
    <div class="sidebar__brand">RTSL</div>

    <nav class="sidebar__nav">
      <RouterLink class="sidebar__link" to="/">Home</RouterLink>
      <RouterLink class="sidebar__link" to="/camera">Camera</RouterLink>
      <RouterLink class="sidebar__link" to="/about">About</RouterLink>
    </nav>
  </aside>
</template>

<style scoped>
/* Hidden by default (phones/tablets) */
.sidebar {
  display: none;
}

/* Show on laptop+ */
@media (min-width: 1024px) {
  .sidebar {
    display: block;
    width: 260px;
    height: 100vh;
    position: sticky;
    top: 0;

    padding: 1.25rem 1rem;

    /* DARKER BLUE SIDEBAR */
    background: #7FB6E3;

    border-right: 1px solid rgba(31, 45, 61, 0.25);
  }

  .sidebar__brand {
    font-weight: 800;
    font-size: 1.1rem;
    margin-bottom: 1rem;
    color: #1f2d3d;
  }

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
    background: rgba(31, 45, 61, 0.10);
  }

  /* Active route */
  .sidebar__link.router-link-active {
    background: rgba(31, 45, 61, 0.20);
    font-weight: 700;
  }
}
</style>
