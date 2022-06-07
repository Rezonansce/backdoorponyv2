<template>
  <div class="">
    <v-row class="flex-nowrap">
      <v-col cols="3" class="d-flex align-center">
        <div class="text-h4 text-left ml-2">
          {{ metric.pretty_name }}
        </div>
        <v-spacer></v-spacer>
        <information
          :actionName="metric.pretty_name"
          :infoText="metric.info"
          :hasLink="false"
        />
      </v-col>
      <v-divider
        class="my-2 mr-3"
        vertical
        color="white"
      >
      </v-divider>
      <v-col cols="8" class="d-flex flex-row align-center scrollable">
        <template
          v-for="(plot, i) in plots"
        >
          <plot-card
            :key="plot"
            :cardKey="i"
            :plot="plot"
            :metric="metric"
            @delete="deletePlot"
            @edited="editPlot"
          />
        </template>
      </v-col>
      <v-col cols="1" class="d-flex flex-row align-center">
        <v-btn
          icon
          color="accent"
          large
          @click="plotDialogDisplay = true"
        >
          <v-icon
            x-large
          >
            mdi-plus
          </v-icon>
        </v-btn>
        <v-dialog
          width="500"
          v-model="plotDialogDisplay"
          class="dialog"
        >
          <template>
            <plot-dialog
              :metric="metric"
              @done="handleAddPlot"
            />
          </template>
        </v-dialog>
      </v-col>
    </v-row>
  </div>
</template>

<style scoped>
.scrollable {
   overflow-x: auto;
   max-width: 90%;
}
</style>

<script>
import Information from './Information.vue';
import PlotCard from './PlotCard.vue';
import PlotDialog from './PlotDialog.vue';

export default {
  components: { Information, PlotCard, PlotDialog },
  setup() {

  },
  props: {
    metric: Object,
    attackKeys: Array,
  },
  data() {
    return {
      plots: [],
      plotDialogDisplay: false,
    };
  },
  methods: {
    handleAddPlot(plot) {
      this.plotDialogDisplay = false;
      this.plots.push(plot);
      this.$emit('modified', this.plots, this.metric.pretty_name);
    },
    deletePlot(key) {
      this.$delete(this.plots, key);
      this.$emit('modified', this.plots, this.metric.pretty_name);
    },
    editPlot(plot, key) {
      this.plots[key] = plot;
      this.$emit('modified', this.plots, this.metric.pretty_name);
    },
  },
  created() {
    console.log(this.attackKeys);
  },
};
</script>
