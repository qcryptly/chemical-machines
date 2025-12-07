import { createRouter, createWebHistory } from 'vue-router'
import NotebookList from '../views/NotebookList.vue'
import NotebookView from '../views/NotebookView.vue'
import VisualizerView from '../views/VisualizerView.vue'

const routes = [
  {
    path: '/',
    name: 'NotebookList',
    component: NotebookList
  },
  {
    path: '/notebook/:id',
    name: 'NotebookView',
    component: NotebookView
  },
  {
    path: '/visualize',
    name: 'Visualize',
    component: VisualizerView
  }
]

const router = createRouter({
  history: createWebHistory(),
  routes
})

export default router
