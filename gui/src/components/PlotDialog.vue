<template>
  <div>
    <v-card
      color="tertiary"
      class="dialog"
    >
      <v-card-title class="text-h5 justify-center">
        Plot for {{ metric.pretty_name }}
      </v-card-title>
      <v-row class="align-content-center">
          <v-card-text class="mt-4  mx-5 text-body-1 justify-center">
            Parameter to use for the x-axis:
          </v-card-text>
          <v-select
            class="mx-10"
            color="accent"
            :items="paramsValues"
            item-text="pretty_name"
            item-value="name"
            :item-disabled="checkIsAxisDisabled"
            v-model="selectedAxis"
            item-color="white"
          ></v-select>
      </v-row>
      <v-row class="align-content-center">
          <v-card-text class="mt-4  mx-5 text-body-1 justify-center">
            Parameter to use for different plots:
          </v-card-text>
          <v-select
            class="mx-10"
            color="accent"
            :items="paramsValues"
            item-text="pretty_name"
            item-value="name"
            :item-disabled="checkIsPlotDisabled"
            v-model="selectedPlot"
            item-color="white"
          ></v-select>
      </v-row>
      <v-row
        class="align-content-center">
          <v-card-text class="mt-4  mx-5 text-body-1 justify-center">
            Set constant values for remaining parameters:
          </v-card-text>
          <template

            v-for="(param, i) in unselectedParams"
          >
            <v-col cols="6" :key="i">
              <v-card-subtitle :key="i" class="mx-6">
              {{ param.pretty_name }} :
            </v-card-subtitle>
            </v-col>
            <v-col cols="6" :key="i">
              <v-select
                :key="i"
                class="mx-10"
                color="accent"
                v-model="constants[param.name]"
                :items="values[param.name]"
                item-color="white"
                @change="handleConstant($event, param.name, param.pretty_name)"
              ></v-select>
            </v-col>
          </template>
      </v-row>
      <v-row class="align-content-center">
        <v-col cols="9">
          <v-card-text class="mt-4  mx-5 text-body-1 justify-center">
            Run the defence after the attack (but before the metrics)
          </v-card-text>
        </v-col>
        <v-col cols="3">
          <v-checkbox
            v-model="isDefended"
            color="accent"
            class="mt-10 justify-center"
            @change="handleChange"
            :disabled="!(metric.is_attack_metric !== undefined
            && metric.is_defence_metric !== undefined)"
          >
          </v-checkbox>
        </v-col>
      </v-row>
      <v-card-actions>
        <v-btn
          v-if="isEdit"
          color="accent"
          text
          @click="deletePlot"
        >
          Delete
        </v-btn>
        <v-spacer></v-spacer>
        <v-btn
          v-if="!isEdit"
          color="accent"
          text
          @click="handleAdd"
        >
          Add
        </v-btn>
        <v-btn
          v-if="isEdit"
          color="accent"
          text
          @click="handleAdd"
        >
          Save changes
        </v-btn>
      </v-card-actions>
    </v-card>
    <v-dialog
        v-model="dialogError"
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
            @click="dialogError = false"
          >
            OK
          </v-btn>
        </v-card-actions>
      </v-card>
    </v-dialog>
  </div>
</template>

<style scoped>
.dialog {
  overflow-x: hidden;
}
html { overflow-x: auto }
</style>

<script>
import PlotServices from '../services/PlotServices';

export default {
  setup() {

  },
  data() {
    return {
      selectedAxis: '',
      selectedPlot: '',
      isDefended: true,
      paramsValues: [],
      dialogError: false,
      errorMessage: '',
      values: {},
      constants: {},
      attackParams: [],
      paramsCopy: [],
      defenceParams: [],
    };
  },
  props: {
    isEdit: Boolean,
    metric: Object,
    editPlot: Object,
  },
  computed: {
    unselectedParams() {
      return this.paramsValues.filter(
        (el) => el.name !== this.selectedAxis && el.name !== this.selectedPlot,
      );
    },
  },
  methods: {
    handleConstant(event, name, prettyName) {
      this.constants[name] = { value: event, pretty_name: prettyName };
    },
    handleAdd() {
      if (!this.selectedAxis) {
        this.errorMessage = 'Please select a parameter to use for the x axis';
        this.dialogError = true;
        return;
      }
      if (!this.selectedPlot) {
        this.errorMessage = 'Please select a parameter to use for the different plots';
        this.dialogError = true;
        return;
      }
      const constObj = {};
      this.unselectedParams.forEach((el) => {
        constObj[el.name] = this.constants[el.name];
      });
      const plotName = this.paramsValues.find((el) => el.name === this.selectedPlot).pretty_name;
      const axisName = this.paramsValues.find((el) => el.name === this.selectedAxis).pretty_name;
      if (this.dialogError === true) {
        return;
      }
      const plot = {
        metric: { pretty_name: this.metric.pretty_name, name: this.metric.name },
        plot: { name: this.selectedPlot, pretty_name: plotName },
        x_axis: { name: this.selectedAxis, pretty_name: axisName },
        is_defended: this.isDefended,
        constants: constObj,
      };
      console.log(plot);
      this.$emit('done', plot);
    },
    deletePlot() {
      this.$emit('delete');
    },
    handleChange() {
      if (this.isDefended) {
        this.paramsValues = this.paramsCopy;
      }
      if (!this.isDefended) {
        this.paramsValues = this.attackParams;
      }
    },
    checkIsPlotDisabled(item) {
      return this.selectedAxis === item.name;
    },
    checkIsAxisDisabled(item) {
      return this.selectedPlot === item.name;
    },
  },
  async created() {
    this.isDefended = true;
    if (this.metric.is_attack_metric !== undefined && this.metric.is_defence_metric === undefined) {
      this.isDefended = false;
    }
    const attackParamsResp = await PlotServices.getAttackParams();
    const defenceParamsResp = await PlotServices.getDefenceParams();
    const values = {};
    const attackParams = [];
    const defenceParams = [];
    Object.entries(attackParamsResp).forEach(([key, entry]) => {
      values[key] = entry.value;
      attackParams.push({ name: key, pretty_name: entry.pretty_name });
    });
    Object.entries(defenceParamsResp).forEach(([key, entry]) => {
      values[key] = entry.value;
      defenceParams.push({ name: key, pretty_name: entry.pretty_name });
    });
    const allParams = attackParams.concat(defenceParams);
    this.paramsValues = allParams;
    this.paramsCopy = allParams;
    this.attackParams = attackParams;
    this.defenceParams = defenceParams;
    this.values = values;
    if (this.isEdit) {
      console.log(this.values);
      console.log(this.editPlot.constants);
      this.selectedAxis = this.editPlot.x_axis.name;
      this.selectedPlot = this.editPlot.plot.name;
      this.isDefended = this.editPlot.is_defended;
      this.constants = this.editPlot.constants;
    }
  },
};
</script>
