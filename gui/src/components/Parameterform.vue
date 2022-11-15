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
      isValid: true,
    };
  },
  props: {
    paramName: String,
    defaultValue: Array,
    isDisabled: Boolean,
    paramKey: String,
    info: String,
  },
  methods: {
    handleInput(input) {
      let changed = 0;
      this.parameter = input.split(',');
      if (input === '' && this.isValid) {
        changed = 1;
        this.isValid = false;
      } else if (input !== '' && !this.isValid) {
        changed = -1;
        this.isValid = true;
      }
      this.$emit('paramChanged', this.paramName, this.parameter, this.paramKey, changed);
    },
  },
};
</script>
