<template>
  <div class="">
    <v-container style="height:100%" class="">
        <v-row style="height:95%">
          <v-col cols="6">
            <div class="text-h3 text-center">
              Datasets
            </div>
            <v-text-field
              label=""
              color="white"
              prepend-inner-icon="mdi-magnify"
              append-outer-icon="mdi-filter"
              class="mx-10 mt-5"
              v-model="dataSearchTerm"
              >
            </v-text-field>
            <div class="mt-6">
              <v-list rounded style="background-color: #1A1A2E;">
                <v-list-item-group
                  v-model="selectedItem"
                  color="white"
                >
                  <v-list-item
                    v-for="dataset in filteredDatasets"
                    :key="dataset.name"
                    @click='handleSelect(dataset, "data")'
                  >
                    <v-list-item-icon>
                      <v-icon color="white" class="mx-4" v-if="dataset.type == 'image'">
                        mdi-image-multiple
                      </v-icon>
                      <v-icon color="white" class="mx-4" v-else-if="dataset.type == 'audio'">
                        mdi-music-box-multiple-outline
                      </v-icon>
                      <v-icon color="white" class="mx-4" v-else-if="dataset.type == 'graph'">
                        mdi-graphql
                      </v-icon>
                      <v-icon color="white" class="mx-4" v-else-if="dataset.type == 'text'">
                        mdi-text-box-multiple-outline
                      </v-icon>
                    </v-list-item-icon>
                    <v-list-item-content>
                      <v-list-item-title v-text="dataset.name"></v-list-item-title>
                    </v-list-item-content>
                    <v-list-item-action
                      class=""
                      style="width: 10%"
                    >
                      <information
                        :actionName="dataset.name.toUpperCase()"
                        :link="dataset.link"
                        :infoText="dataset.info"
                        :hasLink="true"
                      />
                    </v-list-item-action>
                  </v-list-item>
                </v-list-item-group>
              </v-list>
            </div>
          </v-col>
          <v-divider vertical class="my-4" color="white"></v-divider>
          <v-col cols= "6" class="">
            <div class="text-h3 text-center">
              Neural Network
            </div>
            <v-radio-group class="ml-10 mt-16" mandatory>
              <v-radio
                label="Use built-in network"
                color="accent"
                @change="builtin = true"
              >
              </v-radio>
              <v-radio
                label="Upload a neural network"
                color="accent"
                @change="builtin = false"
              >
              </v-radio>
            </v-radio-group>
            <template v-if="builtin === false">
              <upload-card
                :title="'Upload a neural network'"
                :subtitle="
                'This should be a PyTorch model (.pth) and should' +
                ' work with the selected dataset.'"
                :type="'nn'"
                @selected="handleSelect"
                @clear="clear(type)"
                :isDisabled="false"
                class="mx-16 mt-10"
                :acceptTypes="'.pth'"
              />
            </template>
            <v-spacer></v-spacer>
            <div class="text-h1 text-center" align="end">
              <v-btn
                color="accent"
                class="rounded-xl"
                min-width="120"
                @click='upload()'
              >
                Continue
              </v-btn>
            </div>

          </v-col>
        </v-row>
    </v-container>
    <v-dialog
        v-model="selectionDialog"
        max-width="260"
        min_height="300"
      >
      <v-card color="tertiary">
        <v-card-title class="headline justify-center" color="">
          Error
        </v-card-title>
        <v-card-subtitle class="justify-center text-center mt-6">
          {{ message }}
        </v-card-subtitle>
        <v-card-actions>
          <v-spacer></v-spacer>
          <v-btn
            color="white"
            text
            @click="selectionDialog = false"
          >
            OK
          </v-btn>
        </v-card-actions>
      </v-card>
    </v-dialog>
    <v-dialog
        v-model="uploading"
        max-width="260"
        persistent
        min_height="300"
      >
      <v-card color="tertiary">
        <v-card-title class="headline justify-center" color="">
          Processing
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
  </div>
</template>

<script>
import UploadService from '../services/UploadFilesService';
import Information from './Information.vue';
import UploadCard from './UploadCard.vue';

export default {
  components: { Information, UploadCard },
  setup() {},
  data() {
    return {
      dialog: false,
      selectedNeuralNetwork: '',
      selectedDataSet: '',
      message: '',
      selectedType: '',
      selectionDialog: false,
      fileInfos: [],
      datasets: [],
      dataSearchTerm: '',
      builtin: true,
      uploading: false,
      selectedItem: '',
    };
  },
  computed: {
    filteredDatasets() {
      return this.datasets.filter(
        (dataset) => dataset.name
          .toString().toLowerCase().includes(this.dataSearchTerm.toLowerCase()),
      );
    },
  },
  methods: {
    handleSelect(file, type) {
      this.active = '';
      if (type === 'nn') {
        this.selectedNeuralNetwork = file;
      }
      if (type === 'data') {
        this.selectedType = file.type;
        this.selectedDataSet = file.name;
      }
    },
    async upload() {
      if (!this.selectedNeuralNetwork && !this.builtin) {
        this.message = 'Please select a Neural Network!';
        this.selectionDialog = true;
        return;
      }
      if (!this.selectedDataSet) {
        this.message = 'Please select a dataset';
        this.selectionDialog = true;
        return;
      }

      if (!this.builtin) {
        if (this.selectedNeuralNetwork.name.split('.').pop() !== 'pth') {
          this.message = 'The model should be a .pth file';
          this.selectionDialog = true;
          return;
        }
      }

      this.selectionDialog = false;

      if (this.builtin) this.selectedNeuralNetwork = '';

      this.message = '';
      this.uploading = true;
      await UploadService.upload(this.selectedType, this.selectedDataSet,
        this.selectedNeuralNetwork)
        .catch(() => {
          this.uploading = false;
          this.message = 'Could not upload the file';
        });
      if (this.message === '') {
        this.uploading = false;
        this.$emit('handleClick');
      }
    },
  },
  async created() {
    const data = await UploadService.getDataSets();
    const dataArr = [];
    Object.entries(data).forEach((group, key) => {
      if (Object.values(group[1]).length > 0) {
        Object.values(group[1]).forEach((dataset) => {
          dataArr.push({
            name: dataset.pretty_name,
            info: dataset.info,
            link: dataset.link,
            type: Object.keys(data)[key],
          });
        });
      }
    });
    this.datasets = dataArr.flat();
  },
};
</script>
