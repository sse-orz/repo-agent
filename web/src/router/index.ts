import { createRouter, createWebHistory } from 'vue-router'
import Home from '../pages/Home.vue'
import RepoDetail from '../pages/RepoDetail.vue'

const routes = [
  { path: '/', name: 'Home', component: Home },
  { path: '/detail/:repoId?', name: 'RepoDetail', component: RepoDetail },
]

const router = createRouter({
  history: createWebHistory(),
  routes,
})

export default router
