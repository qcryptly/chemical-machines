import { createRouter, createWebHistory } from 'vue-router'
import NotebookList from '../views/NotebookList.vue'
import WorkspaceView from '../views/WorkspaceView.vue'

const routes = [
  {
    path: '/',
    name: 'NotebookList',
    component: NotebookList
  },
  {
    path: '/notebook/:id',
    name: 'Workspace',
    component: WorkspaceView
  },
  {
    path: '/workspace',
    name: 'NewWorkspace',
    component: WorkspaceView
  }
]

const router = createRouter({
  history: createWebHistory(),
  routes
})

export default router
