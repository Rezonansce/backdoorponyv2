<template>
  <div class="">
    <v-card color="tertiary" rounded="xl" min-width="">
      <v-card-title class="text-h5 justify-center text-center">
        <strong>{{title}}</strong>
      </v-card-title>
      <v-card-actions class="justify-center align-content-center">
        <v-file-input
          label="File input"
          outlined
          dense
          color="accent accent"
          style="width= 60%"
          class="mx-10"
          :disabled="isDisabled == true"
          :accept="acceptTypes"
          @change="handleFileUpload"
          @click:clear="clearNetwork"
        ></v-file-input>
      </v-card-actions>
      <v-card-subtitle class="justify-center text-center">
        {{subtitle}}
      </v-card-subtitle>
    </v-card>
  </div>
</template>

<script>
export default {
  props: {
    title: String,
    subtitle: String,
    type: String,
    isDisabled: Boolean,
    acceptTypes: String,
  },
  data() {
    return {
      file: '',
    };
  },
  methods: {
    handleFileUpload(file) {
      this.file = file;
      this.$emit('selected', this.file, this.type);
    },
    clearNetwork() {
      this.$emit('clear');
    },
  },
};
</script>
