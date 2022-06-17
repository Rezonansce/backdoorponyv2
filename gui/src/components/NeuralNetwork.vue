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
                  @click='handleSelectClick(dataset, "data")'
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
                    <v-list-item-title v-text="dataset.name.toUpperCase()"></v-list-item-title>
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
        <v-divider
          class="mx-2"
          vertical
          color="white"
        ></v-divider>
        <v-col>
          <div class="text-h3 text-center" style="height: 50px">
            Hyperparameters
          </div>
          <div class="text-h6 text-left ml-4 mt-8" style="height: 30px">
            {{ selectedModel.toUpperCase() }}
          </div>
          <v-list
            class="overflow-y-auto primary mx-0 mt-1 mb-3"
            :height="paramHeight"
          >
            <template v-for="(param, i) in modelParams">
              <v-list-item :key="i" class="">
                <parameter
                  :paramName="param.pretty_name"
                  :defaultValue="param.value"
                  class="mt-1"
                  :paramKey="i"
                  :info="info[i]"
                  @paramChanged="updateModelParams"
                />
              </v-list-item>
            </template>
          </v-list>
          <!--            <v-divider color="white"></v-divider>-->
          <div class="mt-8 text-center">
            <v-btn
              color="accent"
              class="rounded-xl"
              min-width="120"
              @click='upload()'>
              Continue
            </v-btn>
            <v-dialog
              v-model="executeError"
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
                    @click="executeError = false"
                  >
                    OK
                  </v-btn>
                </v-card-actions>
              </v-card>
            </v-dialog>
            <v-dialog
              v-model="training"
              max-width="260"
              persistent
              min_height="300"
            >
              <v-card color="tertiary">
                <v-card-title class="headline justify-center" color="">
                  Training
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
                  This may take a while, training a neural network is computationally intensive.
                </v-card-subtitle>
              </v-card>
            </v-dialog>
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
import ModelService from '../services/ModelService';
import Parameter from './Parameter.vue';
import Information from './Information.vue';

export default {
  name: 'app',
  components: { Parameter, Information },
  setup() {
  },
  data() {
    return {
      models: [{
        name: 'None',
        info: 'Select this option if you do not want to execute a model.',
        input_type: 'none',
      }],
      selectedModel: '',
      selectedModelItem: '',
      modelSearchTerm: '',
      modelCategory: '',
      modelParams: '',
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
      types: {},
      info: {},
    };
  },
  computed: {
    height() {
      switch (this.$vuetify.breakpoint.name) {
        case 'xs':
          return 220;
        case 'sm':
          return 400;
        case 'md':
          return 300;
        case 'lg':
          return 440;
        case 'xl':
          return 620;
        default:
          return 450;
      }
    },
    paramHeight() {
      switch (this.$vuetify.breakpoint.name) {
        case 'xs':
          return 220;
        case 'sm':
          return 400;
        case 'md':
          return 300;
        case 'lg':
          return 145;
        case 'xl':
          return 220;
        default:
          return 450;
      }
    },
    filteredModels() {
      return this.models.filter(
        (model) => model.name.toLowerCase().includes(this.modelSearchTerm.toLowerCase()),
      );
    },
    filteredDatasets() {
      return this.datasets.filter(
        (dataset) => dataset.name
          .toString().toLowerCase().includes(this.dataSearchTerm.toLowerCase()),
      );
    },
  },
  methods: {
    async handleSelectClick(file, type) {
      if (type === 'data') {
        this.selectedType = file.type;
        this.selectedDataSet = file.name;
      }
      this.isModel = false;
      this.selectedModel = '';
      this.modelParams = {};
      if (file.name !== 'None') {
        this.isModel = true;
        this.selectedModel = file.name;
        const params = await ModelService.getModelParams(file.name);
        this.modelParams = params[0].defaults;
        this.modelCategory = params[0].category;
        Object.entries(params[0].defaults).forEach(([key, entry]) => {
          this.types[key] = entry.default_value[0].constructor;
          this.info[key] = entry.info;
          this.updateModelParams(entry.pretty_name, entry.default_value, key);
        });
      }
      if (this.selectedModel) this.notReady = false;
    },
    // async handleExecuteClick() {
    //   this.executeError = false;
    //   this.errorMessage = '';
    //   this.executing = true;
    //   const resultData = await ModelService.execute(
    //     this.isModel,
    //     JSON.stringify(this.selectedModel),
    //     JSON.stringify(this.modelParams),
    //     JSON.stringify(this.modelCategory),
    //   ).catch(() => {
    //     this.executing = false;
    //     this.executeError = true;
    //     this.errorMessage = 'An error occured';
    //   });
    //   ModelService.setResponse(resultData);
    //   if (!this.errorMessage) {
    //     this.executing = false;
    //     this.$emit('handleClick');
    //   }
    //   this.$emit('executed');
    // },
    updateModelParams(name, newValues, paramKey, info) {
      const convertValues = this.convert(newValues, paramKey);
      this.modelParams[paramKey] = { pretty_name: name, value: convertValues, info };
    },
    convert(values, paramKey) {
      const type = this.types[paramKey];
      return values.map(type);
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
      this.executeError = false;
      this.errorMessage = '';
      this.uploading = true;
      await ModelService.selectModel(
        this.selectedType,
        this.selectedDataSet,
        JSON.stringify(this.modelParams),
      ).catch(() => {
        this.uploading = false;
        this.executeError = true;
        this.errorMessage = 'An error occured';
      }).then(() => {
        if (!this.errorMessage) {
          this.uploading = false;
          this.$root.$refs.type = this.selectedType;
          this.$root.$refs.AttacksAndDefences.filterAttacksAndDefences();
          this.$emit('handleClick');
        }
      });

      this.selectionDialog = false;

      if (this.builtin) this.selectedNeuralNetwork = '';
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
