<script setup lang="ts">
import "@/screens/style/about.css";
import logo from "@/assets/logo_without_text-removebg-preview.png";
import Sidebar from "@/components/Sidebar.vue";
// import PageWithSidebar from "@/components/PageWithSidebar.vue";
</script>

<template>
  <div style="padding: 2rem" class="about-wrapper">
    <aside class="sidebar">
      <div>
        <span class="sidebar-title">About Us</span>
        <Sidebar />
      </div>
      <div class="logo">
        <img :src="logo" alt="RTSL Logo" class="about-sidebar-logo" />
        <h1 class="sidebar-title">RTSL</h1>
        <p class="sidebar-subtitle">Real-Time Sign Language</p>
      </div>
    </aside>
    <main>
      <div class="about-box">
        <h1 class="about-title">About Us</h1>
        <p class="about-text">
          From the onset of the project, our team wanted to create a tool that
          would help people. In our research for topics and data, we found
          datasets related to ASL and decided that we wanted to try to create an
          application that can translate ASL to English. This would
          significantly aid in the communication between people who use ASL and
          those who do not. <br />
          <br />
          There are two ways to use RTSL, you can use our "Camera Translate"
          option to use the camera to translate ASL, or you can download our web
          extension that will translate ASL to English in any calls you are in.
        </p>
        <img :src="logo" alt="RTSL Logo" class="about-logo" />
        <h1 class="home-title">RTSL</h1>
      </div>
    </main>
  </div>
</template>
<!--
<template>
  <PageWithSidebar>
    <div class="about-wrapper">
      <div class="about-box">
        <h1 class="about-title">About Us</h1>

        <p class="about-text">
          From the onset of the project, our team wanted to create a tool that would help people.
          In our research for topics and data, we found datasets related to ASL and decided that we wanted to try to create an application that can translate ASL to English.
          This would significantly aid in the communication between people who use ASL and those who do not.
          <br /><br />
          There are two ways to use RTSL, you can use our "Camera Translate" option to use the camera to translate ASL, or you can download our web extension that will translate ASL to English in any calls you are in.
        </p>

        <img :src="logo" alt="RTSL Logo" class="about-logo" />
        <h2 class="home-title">RTSL</h2>
      </div>
    </div>
  </PageWithSidebar>
</template>
-->
