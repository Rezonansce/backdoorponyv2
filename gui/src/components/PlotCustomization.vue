<template>
  <div class="plotCustomization">
    <v-container class="">
      <v-divider
      class="my-4"

      color="white"
      ></v-divider>
      <template v-for="(metric, i) in metrics">
        <plot-creator
          :key="'plot-creator-' + i"
          :metric="metric"
          @modified="handleModified"
          :attackKeys="attackKeys"
        />
        <v-divider
          class="my-4"
          color="white"
          :key="'divider' + i"
        ></v-divider>
      </template>
      <v-row class="mt-6">
        <v-spacer></v-spacer>
        <v-btn
          color="accent"
          class="rounded-xl"
          min-width="120"
          @click="executePlotting"
        >
          Plot
        </v-btn>
      </v-row>
    </v-container>
    <v-dialog
        v-model="uploading"
        max-width="260"
        persistent
        min_height="300"
      >
      <v-card color="tertiary">
        <v-card-title class="headline justify-center" color="">
          Plotting
        </v-card-title>
        <v-card-text class="justify-center align-content-center text-center">
          <v-progress-circular
            :size="50"
            indeterminate
            color="accent"
            class="justify-center"
          ></v-progress-circular>
        </v-card-text>
        <v-card-subtitle class="justify-center text-center">
          This may take a while.
        </v-card-subtitle>
      </v-card>
    </v-dialog>
    <v-dialog
        v-model="plotError"
        max-width="290"
      >
      <v-card color="tertiary">
        <v-card-title class="headline" color="">
          Error
        </v-card-title>
        <v-card-text>
          {{ errorMessage }}
        </v-card-text>
        <v-card-actions>
          <v-spacer></v-spacer>
          <v-btn
            color="white"
            text
            @click="plotError = false"
          >
            OK
          </v-btn>
        </v-card-actions>
      </v-card>
    </v-dialog>
  </div>
</template>

<style scoped>
.plotCustomization {
  flex: 1;
  overflow-y: auto; /* it works! */
  overflow-x: hidden;
}
</style>

<script>
import PlotCreator from './PlotCreator.vue';
import PlotServices from '../services/PlotServices';

export default {
  components: { PlotCreator },
  setup() {

  },
  data() {
    return {
      metrics: {},
      plots: [],
      error: false,
      uploading: false,
      plotError: false,
      errorMessage: '',
    };
  },
  async created() {
    const metrics = await PlotServices.getMetrics();
    this.metrics = metrics;
    Object.values(this.metrics).forEach((metric) => {
      this.plots.push({ metric: metric.pretty_name, plot: [] });
    });
  },
  methods: {
    handleModified(plots, metric) {
      this.plots.find((el) => el.metric === metric).plot = plots;
    },
    async executePlotting() {
      this.plotError = false;
      this.uploading = true;
      const plots = {};
      const final = this.plots.map((el) => el.plot).flat();
      final.forEach((plot, i) => {
        plots[`plot${i}`] = plot;
      });
      const resp = await PlotServices.plot(JSON.stringify(plots))
        .catch((error) => {
          this.errorMessage = 'Error while plotting';
          this.uploading = false;
          this.errorMessage = true;
          console.log(error);
        });
      PlotServices.setResponse(resp);
      this.uploading = false;
      if (!this.plotError) this.$emit('done');
    },
  },
};
</script>
