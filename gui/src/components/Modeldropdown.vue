<template>
  <v-row class="justify-content-center">
    <v-col cols="7">
      <div class="text-subtitle-1">{{paramName}}</div>
    </v-col>
    <v-col cols="3">
      <v-select
      color="accent"
      v-model="parameter"
      :items="values"
      @input="handleInput"
      :single-select=true
      ref="temp"
      >
      <template v-slot:item="{item, on, attrs}">
        <v-list-item v-on="on">
          <v-list-item-action>
            <v-simple-checkbox :value="attrs.inputValue"
              v-on="on"
              color="red"
              :ripple="false">
            </v-simple-checkbox>
          </v-list-item-action>
          <v-list-item-content color="green">
            {{ item }}
          </v-list-item-content>
        </v-list-item>
      </template>
    </v-select>
    </v-col>
    <v-col cols="2">
      <information
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

    .theme--light.v-application {
      background-color: #432032;
    }

    .theme--light.v-list {
      background: #5b3648;
    }

    .v-list-item__content {
      color: white;
    }

    .theme--light.v-list-item:hover:before {
      opacity: 0.14;
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
      parameter: this.defaultValue[0],
      parameters: this.values,
      selectedItems: [],
      isValid: true,
    };
  },
  props: {
    paramName: String,
    defaultValue: Array,
    values: Array,
    isDisabled: Boolean,
    paramKey: String,
    info: String,
  },
  methods: {
    handleInput(input) {
      const elem = this.$refs.temp;
      elem.click();
      this.parameter = input;
      this.$emit('paramChanged', this.paramName, [this.parameter], this.parameters, this.paramKey, 0);
    },
  },
};
</script>
