<template>
  <div style="height:100%" class="primary">
      <v-stepper
        v-model="e1"
        non-linear
        class="secondary elevation-0"
        style="height:100%"
      >
        <v-stepper-header id="stepper-header">
          <v-stepper-step
            editable
            step="1"
            class="ml-10"
            color="accent"
          >
            <strong class="ml-2 subheading">Home</strong>
          </v-stepper-step>

          <v-divider color="white"></v-divider>

          <v-stepper-step
            editable
            step="2"
            color="accent"
          >
            <strong class="ml-2 subheading">Neural Network</strong>
          </v-stepper-step>

          <v-divider color="white"></v-divider>

          <v-stepper-step
            editable
            step="3"
            color="accent"
          >
            <strong class="ml-2 subheading">Attacks & Defences</strong>
          </v-stepper-step>

          <v-divider color="white"></v-divider>

          <v-stepper-step
            editable
            step="4"
            color="accent"
          >
            <strong class="ml-2 subheading">Plot Customization</strong>
          </v-stepper-step>

          <v-divider color="white"></v-divider>

          <v-stepper-step
            editable
            step="5"
            class="mr-10"
            color="accent"
          >
            <strong class="ml-2 subheading">Results</strong>
          </v-stepper-step>
        </v-stepper-header>
        <v-stepper-items class="primary" style="height:100%">
          <v-stepper-content step="1" >
            <home @handleClick="nextStep(1)"/>
          </v-stepper-content>
          <v-stepper-content step="2" style="height:100%">
            <neural-network
              style="height: 78vh"
              @handleClick="nextStep(2)"/>
          </v-stepper-content>
          <v-stepper-content step="3" class="" style="height:100%">
            <attacks-and-defences
              @handleClick="nextStep(3)"
              style="height: 78vh">
            </attacks-and-defences>
          </v-stepper-content>
          <v-stepper-content step="4" class="">
            <plot-customization
              v-if="e1 >= 4"
              style="height: 78vh"
              @done="nextStep(4)"
            />
          </v-stepper-content>
          <v-stepper-content step="5" class="" style="height:100%">
            <results v-if="e1 >= 5" style="height: 78vh"/>
          </v-stepper-content>
        </v-stepper-items>

      </v-stepper>
  </div>
</template>

<script>
import axios from 'axios';
import Home from './Home.vue';
import NeuralNetwork from './NeuralNetwork.vue';
import AttacksAndDefences from './AttacksAndDefences.vue';
import PlotCustomization from './PlotCustomization.vue';
import Results from './Results.vue';

export default {
  components: {
    Home,
    NeuralNetwork,
    AttacksAndDefences,
    PlotCustomization,
    Results,
  },
  name: 'BackdoorPony',
  data() {
    return {
      msg: '',
      e1: 1,
      steps: 5,
      resultData: '',
    };
  },
  watch: {
    steps(val) {
      if (this.e1 > val) {
        this.e1 = val;
      }
    },
  },
  methods: {
    getMessage() {
      const path = 'http://localhost:5000/';
      axios
        .get(path)
        .then((res) => {
          this.msg = res.data;
        })
        .catch((error) => {
          // eslint-disable-next-line
          console.error(error);
        });
    },
    nextStep(n) {
      if (n === this.steps) {
        this.e1 = 1;
      } else {
        this.e1 = n + 1;
      }
    },
  },
  created() {
    // this.getMessage();
  },
};
</script>
