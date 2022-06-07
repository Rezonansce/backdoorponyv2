import Vue from 'vue';
import Vuetify from 'vuetify/lib/framework';

Vue.use(Vuetify);

const vuetify = new Vuetify({
  theme: {
    defaultTheme: 'dark',
    themes: {
      dark: {
        background: '#FFFFFF',
        surface: '#FFFFFF',
        primary: '#1A1A2E',
        secondary: '#16213E',
        tertiary: '0f3460',
        accent: '#E94560',
      },
    },
    dark: true,
  },
});

export default vuetify;
