<template>
  <v-row class="justify-content-center">
    <v-col cols="7">
      <div class="text-subtitle-1">{{paramName}}</div>
    </v-col>
    <v-col cols="3">
      <v-text-field label="" color="white" style="" dense
        class="centered-input"
        :value="defaultValue"
        @input="handleInput"
        v-model="parameter"
        :rules="[rules.required, rules.interval]"
      >
      </v-text-field>
    </v-col>
    <v-col cols="2">
      <information
        :actionName="paramName"
        :hasLink="false"
        :infoText="info"
      />
    </v-col>
  </v-row>
</template>

<style scoped>
    .centered-input >>> input {
      text-align: center
    }
</style>

<script>
import Information from './Information.vue';

export default {
  components: { Information },
  setup() {

  },
  data() {
    return {
      parameter: this.defaultValue,
      rules: {
        required: (value) => !!value.toString() || 'At least one value has to be provided.',
        interval: (value) => value.toString().split(',')
          .every((x) => x >= this.minValue && x <= this.maxValue)
          || 'Ensure that every hyperparameter value is within the bounds.',
      },
    };
  },
  props: {
    paramName: String,
    defaultValue: Array,
    isDisabled: Boolean,
    paramKey: String,
    minValue: Number,
    maxValue: Number,
    info: String,
  },
  methods: {
    handleInput(input) {
      this.parameter = input.split(',');
      this.$emit('paramChanged', this.paramName, this.parameter, this.paramKey);
    },
  },
};
</script>
