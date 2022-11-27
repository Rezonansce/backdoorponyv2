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
          <div>
            <v-list rounded style="background-color: #1A1A2E;"
            :height="height" class="overflow-y-auto">
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
          <div class="text-h6 text-left ml-4 mt-8 mb-4" style="height: 30px">
            {{ selectedModel.toUpperCase() }}
          </div>
          <v-list class="overflow-y-auto primary mx-0 mt-1 mb-3"
                  :height="paramHeight">
            <template v-for="(param, i) in modelParamsList">
              <v-list-item :key="i" class="">
                <parameterform :paramName="param.pretty_name"
                               :defaultValue="param.value"
                               class="mt-1"
                               :paramKey="i"
                               :info="infoListModel[i]"
                               @paramChanged="updateModelParamsList" />
              </v-list-item>
            </template>
            <template v-for="(param, i) in modelParamsForm">
              <v-list-item :key="i" class="">
                <modelform :paramName="param.pretty_name"
                               :defaultValue="param.value"
                               class="mt-1"
                               :paramKey="i"
                               :info="infoFormModel[i]"
                               @paramChanged="updateModelParamsForm" />
              </v-list-item>
            </template>
            <template v-for="(param, i) in modelParamsDropdown">
              <v-list-item :key="i" class="">
                <parameterdropdown :paramName="param.pretty_name"
                                   :defaultValue="param.value"
                                   :values="param.values"
                                   class="mt-1"
                                   :paramKey="i"
                                   :info="infoDropdownModel[i]"
                                   @paramChanged="updateModelParamsDropdown" />
              </v-list-item>
            </template>
            <template v-for="(param, i) in modelParamsRange">
              <v-list-item :key="i" class="">
                <parameterrange :paramName="param.pretty_name"
                                :defaultValue="param.value"
                                class="mt-1"
                                :paramKey="i"
                                :minValue="param.minimum"
                                :maxValue="param.maximum"
                                :info="infoRangeModel[i]"
                                @paramChanged="updateModelParamsRange" />
              </v-list-item>
            </template>
          </v-list>
          <!--            <v-divider color="white"></v-divider>-->
          <div class="mt-8 text-center">
            <v-btn
              color="accent"
              class="rounded-xl"
              min-width="120"
              @click='upload()'
              :disabled="isDisabled">
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
import Parameterform from './Parameterform.vue';
import Modelform from './Modelform.vue';
import Parameterdropdown from './Modeldropdown.vue';
import Parameterrange from './Parameterrange.vue';
import Information from './Information.vue';

export default {
  name: 'app',
  components: {
    Modelform, Parameterform, Parameterdropdown, Parameterrange, Information,
  },
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
      modelParamsForm: '',
      modelParamsList: '',
      modelParamsDropdown: '',
      modelParamsRange: '',
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
      typesFormModel: {},
      infoFormModel: {},
      typesListModel: {},
      infoListModel: {},
      typesDropdownModel: {},
      infoDropdownModel: {},
      typesRangeModel: {},
      infoRangeModel: {},
      totalIncorrect: 0,
      isDisabled: false,
    };
  },
  computed: {
    height() {
      switch (this.$vuetify.breakpoint.name) {
        case 'xs':
          return 210;
        case 'sm':
          return 390;
        case 'md':
          return 290;
        case 'lg':
          return 430;
        case 'xl':
          return 610;
        default:
          return 440;
      }
    },
    paramHeight() {
      switch (this.$vuetify.breakpoint.name) {
        case 'xs':
          return 440;
        case 'sm':
          return 800;
        case 'md':
          return 600;
        case 'lg':
          return 290;
        case 'xl':
          return 440;
        default:
          return 900;
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
      this.modelParamsForm = {};
      this.modelParamsDropdown = {};
      this.modelParamsRange = {};
      this.modelParamsList = {};
      if (file.name !== 'None') {
        this.isModel = true;
        this.selectedModel = file.name;
        const params = await ModelService.getModelParams(file.name);
        this.modelParamsForm = params[0].defaults_form;
        this.modelParamsDropdown = params[0].defaults_dropdown;
        this.modelParamsRange = params[0].defaults_range;
        this.modelCategory = params[0].category;
        this.modelParamsList = params[0].defaults_list;
        Object.entries(params[0].defaults_form).forEach(([key, entry]) => {
          this.typesFormModel[key] = entry.default_value[0].constructor;
          this.infoFormModel[key] = entry.info;
          this.updateModelParamsForm(entry.pretty_name, entry.default_value, key, 0);
        });
        Object.entries(params[0].defaults_dropdown).forEach(([key, entry]) => {
          this.typesDropdownModel[key] = entry.default_value[0].constructor;
          this.infoDropdownModel[key] = entry.info;
          this.updateModelParamsDropdown(entry.pretty_name, entry.default_value,
            entry.possible_values, key, 0);
        });
        Object.entries(params[0].defaults_range).forEach(([key, entry]) => {
          this.typesRangeModel[key] = entry.default_value[0].constructor;
          this.infoRangeModel[key] = entry.info;
          this.updateModelParamsRange(entry.pretty_name, entry.default_value, key,
            entry.minimum, entry.maximum, 'setup', true);
        });
        Object.entries(params[0].defaults_list).forEach(([key, entry]) => {
          this.typesListModel[key] = entry.default_value[0].constructor;
          this.infoListModel[key] = entry.info;
          this.updateModelParamsList(entry.pretty_name, entry.default_value, key, 0);
        });
      }
      if (this.selectedModel) this.notReady = false;
      console.log(this.filteredModels);
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
    updateModelParamsForm(name, newValues, paramKey, changed, info) {
      const convertValues = this.convertFormModel(newValues, paramKey);
      this.modelParamsForm[paramKey] = { pretty_name: name, value: convertValues, info };
      this.totalIncorrect += changed;
      this.isDisabled = this.totalIncorrect !== 0;
    },
    updateModelParamsList(name, newValues, paramKey, changed, info) {
      const convertValues = this.convertListModel(newValues, paramKey);
      this.modelParamsList[paramKey] = { pretty_name: name, value: convertValues, info };
      this.totalIncorrect += changed;
      this.isDisabled = this.totalIncorrect !== 0;
    },
    updateModelParamsDropdown(name, newValues, newPossibleValues, paramKey, changed, info) {
      const convertValues = this.convertDropdownModel(newValues, paramKey);
      const convertValuesTemp = this.convertDropdownModel(newPossibleValues, paramKey);
      this.modelParamsDropdown[paramKey] = {
        pretty_name: name,
        value: convertValues,
        values: convertValuesTemp,
        info,
      };
      this.totalIncorrect += changed;
      this.isDisabled = this.totalIncorrect !== 0;
    },
    updateModelParamsRange(name, newValues, paramKey, newMin, newMax, input, setup, info) {
      const convertValues = this.convertRangeModel(newValues, paramKey);
      if (setup) {
        this.modelParamsRange[paramKey].isValid = true;
      }
      const oldValid = this.modelParamsRange[paramKey].isValid;
      this.modelParamsRange[paramKey] = {
        pretty_name: name,
        value: convertValues,
        minimum: newMin,
        maximum: newMax,
        info,
      };
      if ((newValues.length > 1 || !newValues.every((x) => x >= newMin && x <= newMax) || input === '')
        && oldValid) {
        this.modelParamsRange[paramKey].isValid = false;
        this.totalIncorrect += 1;
      } else if (newValues.length === 1 && newValues.every((x) => x >= newMin && x <= newMax) && input !== ''
        && !oldValid) {
        this.modelParamsRange[paramKey].isValid = true;
        this.totalIncorrect += -1;
      } else {
        this.modelParamsRange[paramKey].isValid = oldValid;
      }
      this.isDisabled = this.totalIncorrect !== 0;
    },
    convertFormModel(values, paramKey) {
      const type = this.typesFormModel[paramKey];
      return values.map(type);
    },
    convertListModel(values, paramKey) {
      const type = this.typesListModel[paramKey];
      return values.map(type);
    },
    convertDropdownModel(values, paramKey) {
      const type = this.typesDropdownModel[paramKey];
      return values.map(type);
    },
    convertRangeModel(values, paramKey) {
      const type = this.typesRangeModel[paramKey];
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
        JSON.stringify(this.modelParamsForm),
        JSON.stringify(this.modelParamsDropdown),
        JSON.stringify(this.modelParamsRange),
        JSON.stringify(this.modelParamsList),
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
