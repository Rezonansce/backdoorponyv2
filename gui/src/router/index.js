import Vue from 'vue';
import VueRouter from 'vue-router';
import BackdoorPony from '../components/BackdoorPony.vue';

Vue.use(VueRouter);

const routes = [
  {
    path: '/',
    name: 'BackdoorPony',
    component: BackdoorPony,
  },
];

const router = new VueRouter({
  mode: 'history',
  base: process.env.BASE_URL,
  routes,
});

export default router;
