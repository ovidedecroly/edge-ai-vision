import { createRouter, createWebHashHistory } from 'vue-router'
import AppLayout from '@/layouts/AppLayout.vue'
import LoginView from '@/views/LoginView.vue'
import DashboardView from '@/views/DashboardView.vue'
import EventsView from '@/views/EventsView.vue'
import WhitelistView from '@/views/WhitelistView.vue'
import DevicesGateView from '@/views/DevicesGateView.vue'
import UsersView from '@/views/UsersView.vue'
import SettingsView from '@/views/SettingsView.vue'
import AuditLogView from '@/views/AuditLogView.vue'

const router = createRouter({
  history: createWebHashHistory(),
  routes: [
    {
      path: '/login',
      name: 'login',
      component: LoginView,
      meta: { public: true }
    },
    {
      path: '/',
      component: AppLayout,
      redirect: '/dashboard',
      children: [
        // Operator & Super Admin shared routes
        { path: 'dashboard', name: 'dashboard', component: DashboardView },
        { path: 'events', name: 'events', component: EventsView },
        { path: 'whitelist', name: 'whitelist', component: WhitelistView },

        // Super Admin exclusive routes
        { path: 'devices', name: 'devices', component: DevicesGateView },
        { path: 'users', name: 'users', component: UsersView },
        { path: 'settings', name: 'settings', component: SettingsView },
        { path: 'audit', name: 'audit', component: AuditLogView },

        // Redirect old Soloraya / legacy pages to dashboard
        { path: 'live', redirect: '/dashboard' },
        { path: 'cameras', redirect: '/devices' },
        { path: 'datasets', redirect: '/dashboard' },
        { path: 'models', redirect: '/dashboard' },
        { path: 'experiments', redirect: '/dashboard' },
        { path: 'system', redirect: '/settings' }
      ]
    },
    {
      path: '/:pathMatch(.*)*',
      redirect: '/dashboard'
    }
  ]
})

router.beforeEach((to, _from, next) => {
  const token = localStorage.getItem('access_token')
  if (!to.meta.public && !token) {
    next({ name: 'login' })
  } else if (to.name === 'login' && token) {
    next({ name: 'dashboard' })
  } else {
    next()
  }
})

export default router
