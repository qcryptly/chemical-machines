import { createRouter, createWebHistory } from 'vue-router'
import WorkspaceList from '../views/WorkspaceList.vue'
import WorkspaceView from '../views/WorkspaceView.vue'

const routes = [
  {
    path: '/',
    name: 'WorkspaceList',
    component: WorkspaceList
  },
  {
    path: '/workspace/:id',
    name: 'Workspace',
    component: WorkspaceView
  },
  // Legacy route redirect
  {
    path: '/notebook/:id',
    redirect: to => ({ path: `/workspace/${to.params.id}` })
  }
]

const router = createRouter({
  history: createWebHistory(),
  routes
})

export default router
