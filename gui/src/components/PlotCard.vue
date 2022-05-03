<template>
  <div>
    <v-card
      min-width="250"
      height="120"
      color="tertiary"
      class="mx-3"
      @click="dialog = true"
    >
      <v-card-text class>
        <div class="text--primary">
          X-axis: {{ axisVal }}<br/>
          Plots for: {{ plotVal }}<br/>
          Constants: {{ constantsNames.toString() }}
        </div>
      </v-card-text>
    </v-card>
    <v-dialog
      width="500"
      v-model="dialog"
      height="800"
    >
      <plot-dialog
        :isEdit="true"
        @done="handleEdit"
        @delete="handleDelete"
        :editPlot="plot"
        :metric="metric"
      />
    </v-dialog>
  </div>
</template>

<script>
import PlotDialog from './PlotDialog.vue';

export default {
  components: { PlotDialog },
  setup() {

  },
  data() {
    return {
      dialog: false,
      constantsNames: [],
      axisVal: '',
      plotVal: '',
    };
  },
  props: {
    plot: Object,
    metric: Object,
    cardKey: Number,
  },
  methods: {
    ShowDialog() {
      this.dialog = true;
    },
    handleEdit(plot) {
      this.axisVal = plot.x_axis.pretty_name;
      this.plotVal = plot.plot.pretty_name;
      this.constantsNames = Object.values(plot.constants).map((el) => el.pretty_name);
      this.dialog = false;
      this.$emit('edited', plot, this.cardKey);
    },
    handleDelete() {
      console.log(this.cardKey);
      this.dialog = false;
      this.$emit('delete', this.cardKey);
    },
  },
  created() {
    this.constantsNames = Object.values(this.plot.constants).map((el) => el.pretty_name);
    this.axisVal = this.plot.x_axis.pretty_name;
    this.plotVal = this.plot.plot.pretty_name;
  },
};
</script>
