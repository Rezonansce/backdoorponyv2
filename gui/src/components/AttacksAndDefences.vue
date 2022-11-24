<template >
  <div v-if="!loading" class="">
    <v-row style="height:95%">
      <v-col>
        <div class="text-h3 text-center">
          Attacks
        </div>
        <v-text-field
          label=""
          color="white"
          prepend-inner-icon="mdi-magnify"
          append-outer-icon="mdi-filter"
          class="mx-10 mt-5"
          v-model="attackSearchTerm"
        >
        </v-text-field>
        <div>
          <v-list rounded style="background-color: #1A1A2E;"
            :height="height" class="overflow-y-auto primary">
            <v-list-item-group
              v-model="selectedAttackItem"
              color="white"
            >
              <v-list-item
                v-for="attack in filteredAttacks"
                :key="attack.name"
                @click='handleSelectClick(attack.name, "att")'
              >
                <v-list-item-icon>
                  <v-icon color="white" class="mx-4" v-if="attack.input_type == 'image'">
                    mdi-image-multiple
                  </v-icon>
                  <v-icon color="white" class="mx-4" v-else-if="attack.input_type == 'audio'">
                    mdi-music-box-multiple-outline
                  </v-icon>
                  <v-icon color="white" class="mx-4" v-else-if="attack.input_type == 'graph'">
                    mdi-graphql
                  </v-icon>
                  <v-icon color="white" class="mx-4" v-else-if="attack.input_type == 'text'">
                    mdi-text-box-multiple-outline
                  </v-icon>
                  <v-icon color="white" class="mx-4" v-else-if="attack.input_type == 'none'">
                    mdi-cancel
                  </v-icon>
                </v-list-item-icon>
                <v-list-item-content>
                  <v-list-item-title v-text="attack.name.toUpperCase()"></v-list-item-title>
                </v-list-item-content>
                <v-list-item-action
                  class=""
                  style="width: 10%"
                >
                  <information
                    :actionName="attack.name.toUpperCase()"
                    :link="attack.link"
                    :infoText="attack.info"
                    :hasLink="attack.link != undefined"
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
        <div class="text-h3 text-center">
          Defences
        </div>
        <v-text-field
          label=""
          color="white"
          prepend-inner-icon="mdi-magnify"
          append-outer-icon="mdi-filter"
          class="mx-10 mt-5"
          v-model="defenceSearchTerm"
          >
        </v-text-field>
        <div>
          <v-list rounded style="background-color: #1A1A2E;"
            :height="height" class="overflow-y-auto primary">
            <v-list-item-group
              v-model="selectedDefenceItem"
              color="white"
            >
              <v-list-item
                v-for="defence in filteredDefences"
                :key="defence.name"
                @click='handleSelectClick(defence.name, "def")'
              >
                <v-list-item-icon>
                  <v-icon color="white" class="mx-4" v-if="defence.input_type == 'image'">
                    mdi-image-multiple
                  </v-icon>
                  <v-icon color="white" class="mx-4" v-else-if="defence.input_type == 'audio'">
                    mdi-music-box-multiple-outline
                  </v-icon>
                  <v-icon color="white" class="mx-4" v-else-if="defence.input_type == 'graph'">
                    mdi-graphql
                  </v-icon>
                  <v-icon color="white" class="mx-4" v-else-if="defence.input_type == 'text'">
                    mdi-text-box-multiple-outline
                  </v-icon>
                  <v-icon color="white" class="mx-4" v-else-if="defence.input_type == 'none'">
                    mdi-cancel
                  </v-icon>
                </v-list-item-icon>
                <v-list-item-content>
                  <v-list-item-title v-text="defence.name.toUpperCase()"></v-list-item-title>
                </v-list-item-content>
                <v-list-item-action
                  class=""
                  style="width: 10%"
                >
                  <information
                    :actionName="defence.name.toUpperCase()"
                    :link="defence.link"
                    :infoText="defence.info"
                    :hasLink="defence.link != undefined"
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
          {{selectedAttack.toUpperCase()}}
        </div>
        <v-list
            class="overflow-y-auto primary mx-0 mt-1 mb-3"
            :height="paramHeight"
          >
            <template v-for="(param, i) in attackParamsForm" >
              <v-list-item :key="i" class="">
                <parameterform
                  :paramName="param.pretty_name"
                  :defaultValue="param.value"
                  class="mt-1"
                  :paramKey="i"
                  :info="infoFormAttack[i]"
                  @paramChanged="updateAttackParamsForm"
                />
              </v-list-item>
            </template>
            <template v-for="(param, i) in attackParamsDropdown" >
              <v-list-item :key="i" class="">
                <parameterdropdown
                  :paramName="param.pretty_name"
                  :defaultValue="param.value"
                  :values="param.values"
                  class="mt-1"
                  :paramKey="i"
                  :info="infoDropdownAttack[i]"
                  @paramChanged="updateAttackParamsDropdown"
                />
              </v-list-item>
            </template>
            <template v-for="(param, i) in attackParamsRange">
              <v-list-item :key="i" class="">
                <parameterrange
                  :paramName="param.pretty_name"
                  :defaultValue="param.value"
                  class="mt-1"
                  :paramKey="i"
                  :minValue="param.minimum"
                  :maxValue="param.maximum"
                  :info="infoRangeAttack[i]"
                  @paramChanged="updateAttackParamsRange" />
              </v-list-item>
            </template>
        </v-list>
        <v-divider color="white"></v-divider>
        <div class="text-h6 text-left ml-4 mt-8" style="height: 30px">
          {{selectedDefence.toUpperCase()}}
        </div>
        <v-list
            class="overflow-y-auto primary mx-0 mt-1"
            :height="paramHeight"
        >
          <template v-for="(param, i) in defenceParamsForm" >
              <v-list-item :key="i" class="">
                <parameterform
                  :paramName="param.pretty_name"
                  :defaultValue="param.value"
                  class="mt-1"
                  :paramKey="i"
                  :info="infoFormDefence[i]"
                  @paramChanged="updateDefenceParamsForm"
                />
              </v-list-item>
            </template>
            <template v-for="(param, i) in defenceParamsDropdown" >
              <v-list-item :key="i" class="">
                <parameterdropdown
                  :paramName="param.pretty_name"
                  :defaultValue="param.value"
                  :values="param.values"
                  class="mt-1"
                  :paramKey="i"
                  :info="infoDropdownDefence[i]"
                  @paramChanged="updateDefenceParamsDropdown"
                />
              </v-list-item>
            </template>
            <template v-for="(param, i) in defenceParamsRange">
              <v-list-item :key="i" class="">
                <parameterrange
                  :paramName="param.pretty_name"
                  :defaultValue="param.value"
                  class="mt-1"
                  :paramKey="i"
                  :minValue="param.minimum"
                  :maxValue="param.maximum"
                  :info="infoRangeDefence[i]"
                  @paramChanged="updateDefenceParamsRange" />
              </v-list-item>
            </template>
        </v-list>
        <div class="mt-8 text-center">
          <v-btn
            color="accent"
            class="rounded-xl"
            min-width="120"
            :disabled="isDisabled"
            @click='handleExecuteClick()'>
            Execute
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
              v-model="executing"
              max-width="260"
              persistent
              min_height="300"
            >
            <v-card color="tertiary">
              <v-card-title class="headline justify-center" color="">
                Executing
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
                This may take a while, executing an attack and/or a
                defence is computationally intensive.
              </v-card-subtitle>
            </v-card>
          </v-dialog>
        </div>
      </v-col>
    </v-row>
  </div>
</template>

<script>
import Information from './Information.vue';
import Parameterform from './Parameterform.vue';
import Parameterdropdown from './Parameterdropdown.vue';
import Parameterrange from './Parameterrange.vue';
import AttacksAndDefencesService from '../services/AttacksAndDefencesService';

export default {
  name: 'app',
  components: {
    Parameterform, Parameterdropdown, Parameterrange, Information,
  },
  setup() {

  },
  data() {
    return {
      attacks: [{
        name: 'None',
        is_attack: 'True',
        info: 'Select this option if you do not want to execute an attack.',
        input_type: 'none',
      }],
      defences: [{
        name: 'None',
        is_attack: 'False',
        info: 'Select this option if you do not want to execute a defence.',
        input_type: 'none',
      }],
      selectedAttack: '',
      selectedAttackItem: '',
      selectedDefence: '',
      selectedDefenceItem: '',
      attackSearchTerm: '',
      defenceSearchTerm: '',
      dialog: false,
      attackParams: '',
      defenceParams: '',
      loading: true,
      errorMessage: '',
      executeError: false,
      executing: false,
      notReady: '',
      isDefended: false,
      isAttacked: false,
      attackCategory: '',
      defenceCategory: '',
      typesFormAttack: {},
      infoFormAttack: {},
      typesDropdownAttack: {},
      infoDropdownAttack: {},
      typesRangeAttack: {},
      infoRangeAttack: {},
      typesFormDefence: {},
      infoFormDefence: {},
      typesDropdownDefence: {},
      infoDropdownDefence: {},
      typesRangeDefence: {},
      infoRangeDefence: {},
      totalIncorrect: 0,
      isDisabled: false,
    };
  },
  computed: {
    height() {
      switch (this.$vuetify.breakpoint.name) {
        case 'xs': return 220;
        case 'sm': return 400;
        case 'md': return 300;
        case 'lg': return 440;
        case 'xl': return 620;
        default: return 450;
      }
    },
    paramHeight() {
      switch (this.$vuetify.breakpoint.name) {
        case 'xs': return 220;
        case 'sm': return 400;
        case 'md': return 300;
        case 'lg': return 145;
        case 'xl': return 220;
        default: return 450;
      }
    },
    filteredAttacks() {
      return this.attacks.filter(
        (attack) => attack.name.toLowerCase().includes(this.attackSearchTerm.toLowerCase()),
      );
    },

    filteredDefences() {
      return this.defences.filter(
        (defence) => defence.name.toLowerCase().includes(this.defenceSearchTerm.toLowerCase()),
      );
    },
  },
  methods: {
    async handleSelectClick(name, type) {
      if (type === 'att') {
        this.isAttacked = false;
        this.selectedAttack = '';
        this.attackParamsForm = {};
        this.attackParamsDropdown = {};
        this.attackParamsRange = {};
        if (name !== 'None') {
          this.isAttacked = true;
          this.selectedAttack = name;
          const params = await AttacksAndDefencesService.getAttackParams(name);
          this.attackParamsForm = params[0].defaults_form;
          this.attackParamsDropdown = params[0].defaults_dropdown;
          this.attackParamsRange = params[0].defaults_range;
          this.attackCategory = params[0].category;
          Object.entries(params[0].defaults_form).forEach(([key, entry]) => {
            this.typesFormAttack[key] = entry.default_value[0].constructor;
            this.infoFormAttack[key] = entry.info;
            this.updateAttackParamsForm(entry.pretty_name, entry.default_value, key, 0);
          });
          Object.entries(params[0].defaults_dropdown).forEach(([key, entry]) => {
            this.typesDropdownAttack[key] = entry.default_value[0].constructor;
            this.infoDropdownAttack[key] = entry.info;
            this.updateAttackParamsDropdown(entry.pretty_name, entry.default_value,
              entry.possible_values, key, 0);
          });
          Object.entries(params[0].defaults_range).forEach(([key, entry]) => {
            this.typesRangeAttack[key] = entry.default_value[0].constructor;
            this.infoRangeAttack[key] = entry.info;
            this.updateAttackParamsRange(entry.pretty_name, entry.default_value, key,
              entry.minimum, entry.maximum, 'setup', true);
          });
        }
        if (this.selectedAttack && this.selectedDefence) this.notReady = false;
        this.filterAttacks();
      }
      if (type === 'def') {
        this.isDefended = false;
        this.selectedDefence = '';
        this.defenceParamsForm = {};
        this.defenceParamsDropdown = {};
        this.defenceParamsRange = {};
        if (name !== 'None') {
          this.isDefended = true;
          this.selectedDefence = name;
          const params = await AttacksAndDefencesService.getDefenceParams(name);
          this.defenceParamsForm = params[0].defaults_form;
          this.defenceParamsDropdown = params[0].defaults_dropdown;
          this.defenceParamsRange = params[0].defaults_range;
          this.defenceCategory = params[0].category;
          console.log(params);
          Object.entries(params[0].defaults_form).forEach(([key, entry]) => {
            this.typesFormDefence[key] = entry.default_value[0].constructor;
            this.infoFormDefence[key] = entry.info;
            this.updateDefenceParamsForm(entry.pretty_name, entry.default_value, key, 0);
          });
          Object.entries(params[0].defaults_dropdown).forEach(([key, entry]) => {
            this.typesDropdownDefence[key] = entry.default_value[0].constructor;
            this.infoDropdownDefence[key] = entry.info;
            this.updateDefenceParamsDropdown(entry.pretty_name, entry.default_value,
              entry.possible_values, key, 0);
          });
          Object.entries(params[0].defaults_range).forEach(([key, entry]) => {
            this.typesRangeDefence[key] = entry.default_value[0].constructor;
            this.infoRangeDefence[key] = entry.info;
            this.updateDefenceParamsRange(entry.pretty_name, entry.default_value, key,
              entry.minimum, entry.maximum, 'setup', true);
          });
        }
        if (this.selectedAttack && this.selectedDefence) this.notReady = false;
        this.filterDefences();
      }
    },
    async handleExecuteClick() {
      this.executeError = false;
      this.errorMessage = '';
      this.executing = true;
      const resultData = await AttacksAndDefencesService.execute(
        this.isDefended,
        this.isAttacked,
        JSON.stringify(this.selectedAttack),
        JSON.stringify(this.attackParamsForm),
        JSON.stringify(this.attackParamsDropdown),
        JSON.stringify(this.attackParamsRange),
        JSON.stringify(this.selectedDefence),
        JSON.stringify(this.defenceParamsForm),
        JSON.stringify(this.defenceParamsDropdown),
        JSON.stringify(this.defenceParamsRange),
        JSON.stringify(this.attackCategory),
        JSON.stringify(this.defenceCategory),
      ).catch(() => {
        this.executing = false;
        this.executeError = true;
        this.errorMessage = 'An error occured';
      });
      AttacksAndDefencesService.setResponse(resultData);
      if (!this.errorMessage) {
        this.executing = false;
        this.$emit('handleClick');
      }
      this.$emit('executed');
    },
    updateAttackParamsForm(name, newValues, paramKey, changed, info) {
      const convertValues = this.convertFormAttack(newValues, paramKey);
      this.attackParamsForm[paramKey] = { pretty_name: name, value: convertValues, info };
      this.totalIncorrect += changed;
      this.isDisabled = this.totalIncorrect !== 0;
    },
    updateAttackParamsDropdown(name, newValues, newPossibleValues, paramKey, changed, info) {
      const convertValues = this.convertDropdownAttack(newValues, paramKey);
      const convertValuesTemp = this.convertDropdownAttack(newPossibleValues, paramKey);
      this.attackParamsDropdown[paramKey] = {
        pretty_name: name,
        value: convertValues,
        values: convertValuesTemp,
        info,
      };
      this.totalIncorrect += changed;
      this.isDisabled = this.totalIncorrect !== 0;
    },
    updateAttackParamsRange(name, newValues, paramKey, newMin, newMax, input, setup, info) {
      const convertValues = this.convertRangeAttack(newValues, paramKey);
      if (setup) {
        this.attackParamsRange[paramKey].isValid = true;
      }
      const oldValid = this.attackParamsRange[paramKey].isValid;
      this.attackParamsRange[paramKey] = {
        pretty_name: name,
        value: convertValues,
        minimum: newMin,
        maximum: newMax,
        info,
      };
      if ((!newValues.every((x) => x >= newMin && x <= newMax) || input === '')
        && oldValid) {
        this.attackParamsRange[paramKey].isValid = false;
        this.totalIncorrect += 1;
      } else if (newValues.every((x) => x >= newMin && x <= newMax) && input !== ''
        && !oldValid) {
        this.attackParamsRange[paramKey].isValid = true;
        this.totalIncorrect += -1;
      } else {
        this.attackParamsRange[paramKey].isValid = oldValid;
      }
      this.isDisabled = this.totalIncorrect !== 0;
    },
    updateDefenceParamsForm(name, newValues, paramKey, changed, info) {
      const convertValues = this.convertFormDefence(newValues, paramKey);
      this.defenceParamsForm[paramKey] = { pretty_name: name, value: convertValues, info };
      this.totalIncorrect += changed;
      this.isDisabled = this.totalIncorrect !== 0;
    },
    updateDefenceParamsDropdown(name, newValues, newPossibleValues, paramKey, changed, info) {
      const convertValues = this.convertDropdownDefence(newValues, paramKey);
      const convertValuesTemp = this.convertDropdownDefence(newPossibleValues, paramKey);
      this.defenceParamsDropdown[paramKey] = {
        pretty_name: name,
        value: convertValues,
        values: convertValuesTemp,
        info,
      };
      this.totalIncorrect += changed;
      this.isDisabled = this.totalIncorrect !== 0;
    },
    updateDefenceParamsRange(name, newValues, paramKey, newMin, newMax, input, setup, info) {
      const convertValues = this.convertRangeDefence(newValues, paramKey);
      if (setup) {
        this.defenceParamsRange[paramKey].isValid = true;
      }
      const oldValid = this.defenceParamsRange[paramKey].isValid;
      this.defenceParamsRange[paramKey] = {
        pretty_name: name,
        value: convertValues,
        minimum: newMin,
        maximum: newMax,
        info,
      };
      if ((!newValues.every((x) => x >= newMin && x <= newMax) || input === '')
        && oldValid) {
        this.defenceParamsRange[paramKey].isValid = false;
        this.totalIncorrect += 1;
      } else if (newValues.every((x) => x >= newMin && x <= newMax) && input !== ''
        && !oldValid) {
        this.defenceParamsRange[paramKey].isValid = true;
        this.totalIncorrect += -1;
      } else {
        this.defenceParamsRange[paramKey].isValid = oldValid;
      }
      this.isDisabled = this.totalIncorrect !== 0;
    },
    convertFormAttack(values, paramKey) {
      const type = this.typesFormAttack[paramKey];
      return values.map(type);
    },
    convertDropdownAttack(values, paramKey) {
      const type = this.typesDropdownAttack[paramKey];
      return values.map(type);
    },
    convertRangeAttack(values, paramKey) {
      const type = this.typesRangeAttack[paramKey];
      return values.map(type);
    },
    convertFormDefence(values, paramKey) {
      const type = this.typesFormDefence[paramKey];
      return values.map(type);
    },
    convertDropdownDefence(values, paramKey) {
      const type = this.typesDropdownDefence[paramKey];
      return values.map(type);
    },
    convertRangeDefence(values, paramKey) {
      const type = this.typesRangeDefence[paramKey];
      return values.map(type);
    },
    filterAttacks() {
      this.attacks = this.attacks.filter(
        (attack) => attack.input_type === this.$root.$refs.type || attack.input_type === 'none',
      );
    },
    filterDefences() {
      this.defences = this.defences.filter(
        (defence) => defence.input_type === this.$root.$refs.type || defence.input_type === 'none',
      );
    },
    filterAttacksAndDefences() {
      this.filterAttacks();
      this.filterDefences();
    },
  },
  async created() {
    this.$root.$refs.AttacksAndDefences = this;
    const attacks = await AttacksAndDefencesService.getAttacks();
    Array.prototype.push.apply(this.attacks, attacks);
    const defences = await AttacksAndDefencesService.getDefences();
    Array.prototype.push.apply(this.defences, defences);
    this.loading = false;
  },

};
</script>
