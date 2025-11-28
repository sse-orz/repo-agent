import { createRouter, createWebHistory } from 'vue-router'
import Home from '../pages/Home.vue'
import RepoDetail from '../pages/RepoDetail.vue'
import History from '../pages/History.vue'

const routes = [
  { path: '/', name: 'Home', component: Home },
  { path: '/detail/:repoId?', name: 'RepoDetail', component: RepoDetail },
  { path: '/history', name: 'History', component: History },
]

const router = createRouter({
  history: createWebHistory(),
  routes,
})

export default router
