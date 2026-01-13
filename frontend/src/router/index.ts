import { createRouter, createWebHistory } from 'vue-router'
import Home from '@/screens/Home.vue'
import Camera from '@/screens/Camera.vue'
import WebExtension from '@/screens/WebExtension.vue'
import About from '@/screens/About.vue'

const routes = [
  { path: '/', name: 'Home', component: Home },
  { path: '/camera', name: 'Camera', component: Camera },
  { path: '/web_extension', name: 'WebExtension', component: WebExtension },
  { path: '/about', name: 'About', component: About },
]

export default createRouter({
  history: createWebHistory(),
  routes,
})
