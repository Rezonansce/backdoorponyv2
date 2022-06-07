<template>
    <v-container >
      <v-row justify="space-around" align="center">
        <v-col class="" cols="4" align-items="stretch">
          <div>
            <horse/>
          </div>
        </v-col>
        <v-col class="" cols="4" align-items="stretch">
          <div class="text-h1">
            <strong>Backdoor <br/> Pony</strong>
          </div>
        </v-col>
      </v-row>
      <v-row justify="space-around" align="center">
        <v-col class="" cols="5" align-items="stretch">
          <div class="subtitle-2">
            <strong>BackdoorPony is a powerful framework that allows for
              evaluation of different backdoor attacks and defences on
              neural networks. The framework also offers result
              visualisation and configuration saving functionality.
              The client is the AISyLab research group at TU Delft.
              This project is part of the CSE2000 Software Project
              at TU Delft.</strong>
          </div>
        </v-col>
        <v-col class="" cols="2" align-items="center">
          <div class="text-h1">
            <v-btn
              color="accent"
              x-large
              class="rounded-xl"
              @click='handleClick()'>
              Start
            </v-btn>
          </div>
        </v-col>
      </v-row>
    </v-container>
</template>

<script>
import Horse from './Horse.vue';

export default {
  components: { Horse },
  methods: {
    handleClick() {
      this.$emit('handleClick');
    },
  },
};
</script>
