<template>
  <div  style="overflow: auto" >
    <div class="results d-flex flex-wrap justify-center">
      <template
        v-for="(opt, i) in optList"
      >
        <chart
          :key="i"
          :options="opt"
        />
      </template>
    </div>
    <v-row class="mt-8">
      <v-spacer></v-spacer>
      <v-btn v-on:click="downloadConfigurationFile()" rounded="xl" color="accent" class="mx-8 mb-4">
        Download Configuration
      </v-btn>
    </v-row>
  </div>
</template>

<style scoped>
.result {
  flex: 1;
  overflow-y: scroll; /* it works! */
  overflow-x: hidden;
  flex-direction: row;
}
html { overflow-x: auto }
</style>

<script>
import PlotServices from '../services/PlotServices';
import Chart from './Chart.vue';

export default {
  data() {
    return {
      chartCADOptions: '',
      chartBenignAccuracyPatternOptions: '',
      plots: {},
      optList: [],
    };
  },
  components: {
    Chart,
  },
  methods: {
    mapData(data) {
      return [{ x: data.poison, y: data.accuracy }];
    },
    formatBarChart(array) {
      return array.map((el) => ({ name: `${el.poison * 100}% poison`, points: [{ x: el.poison, y: el.cad }] }));
    },
    formatLine(array) {
      return array.map((el) => this.mapData(el)[0]);
    },
    updateData(data) {
      const optList = [];
      Object.values(data).forEach((el, i) => {
        optList[i] = {
          type: 'line',
          title: { label: { text: `${el.metric} against ${el.x_axis}` } },
          legend: {
            header: `,${el.plot}`,
            template: '%icon,%name',
            maxWidth: 400,
            cellSpacing: 8,
          },
          series: Object.values(el.graph),
        };
      });
      this.optList = optList;
    },
    async downloadConfigurationFile() {
      const conf = await PlotServices.getConfigurationFile();
      console.log(conf);
      let text = '';
      conf.forEach((element) => {
        text += `${element}\n`;
      });
      const a = window.document.createElement('a');
      a.href = window.URL.createObjectURL(new Blob([text], { type: 'text/txt' }));
      a.download = 'backdoorpony_configuration.txt';

      // Append anchor to body.
      document.body.appendChild(a);
      a.click();

      // Remove anchor from body
      document.body.removeChild(a);
    },
  },
  async created() {
    const plots = PlotServices.getResponse();
    this.plots = plots;
    console.log(plots);
    this.updateData(this.plots);
  },
};
</script>

<style>
.columnChart {
  height: 300px;
}
</style>
