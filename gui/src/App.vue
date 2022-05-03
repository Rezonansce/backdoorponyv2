<template>
  <v-app>
    <v-main style="width:100%">
      <router-view/>
    </v-main>
    <custom-footer/>
  </v-app>
</template>

<style>
html {
  overflow: hidden !important;
  scrollbar-width: none;
  -ms-overflow-style: none;
}

html::-webkit-scrollbar {
  width: 0;
  height: 0;
}
</style>

<script>
import CustomFooter from './components/CustomFooter.vue';

export default {
  name: 'App',
  components: { CustomFooter },
  data: () => ({
    //
  }),
};
</script>
