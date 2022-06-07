<template>
      <v-card
        color="tertiary"
        rounded=""
        class="text-center ma-3"
        height="430"
        min-width=""
        style="width: 30%"
      >
        <v-card-title class="justify-center">
          {{ title }}
        </v-card-title>
        <template class="">
          <v-card-text class="justify-center">
            <JSCharting
              :options="options"
              class="type"
              :mutable="true">
            </JSCharting>
          </v-card-text>
        </template>
      </v-card>
</template>

<script>
import JSCharting from 'jscharting-vue';

export default {
  setup() {

  },
  props: {
    options: Object,
    type: String,
    title: String,
  },
  components: {
    JSCharting,
  },
};
</script>
