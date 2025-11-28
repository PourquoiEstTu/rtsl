import { createRouter, createWebHistory } from 'vue-router'
import Home from '@/screens/Home.vue'
import Camera from '@/screens/Camera.vue'

const routes = [
  { path: '/', name: 'Home', component: Home },
  { path: '/camera', name: 'Camera', component: Camera },
]

export default createRouter({
  history: createWebHistory(),
  routes,
})
