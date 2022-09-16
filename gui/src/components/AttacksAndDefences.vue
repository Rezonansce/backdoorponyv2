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
                  :info="infoForm[i]"
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
                  :info="infoDropdown[i]"
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
                  :info="infoRange[i]"
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
          <template v-for="(param, i) in defenceParams" >
            <v-list-item :key="i" class="">
              <parameter
                :paramName="param.pretty_name"
                :defaultValue="param.value"
                class="mt-1"
                :paramKey="i"
                :info="info[i]"
                @paramChanged="updateDefenceParams"
              />
            </v-list-item>
          </template>
        </v-list>
        <div class="mt-8 text-center">
          <v-btn
            color="accent"
            class="rounded-xl"
            min-width="120"
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
      typesForm: {},
      infoForm: {},
      typesDropdown: {},
      infoDropdown: {},
      typesRange: {},
      infoRange: {},
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
            this.typesForm[key] = entry.default_value[0].constructor;
            this.infoForm[key] = entry.info;
            this.updateAttackParamsForm(entry.pretty_name, entry.default_value, key);
          });
          Object.entries(params[0].defaults_dropdown).forEach(([key, entry]) => {
            this.typesDropdown[key] = entry.default_value[0].constructor;
            this.infoDropdown[key] = entry.info;
            this.updateAttackParamsDropdown(entry.pretty_name, entry.default_value,
              entry.possible_values, key);
          });
          Object.entries(params[0].defaults_range).forEach(([key, entry]) => {
            this.typesRange[key] = entry.default_value[0].constructor;
            this.infoRange[key] = entry.info;
            this.updateAttackParamsRange(entry.pretty_name, entry.default_value, key, entry.minimum,
              entry.maximum);
          });
        }
        if (this.selectedAttack && this.selectedDefence) this.notReady = false;
        return;
      }
      if (type === 'def') {
        this.isDefended = false;
        this.selectedDefence = '';
        this.defenceParams = {};
        if (name !== 'None') {
          this.isDefended = true;
          this.selectedDefence = name;
          const params = await AttacksAndDefencesService.getDefenceParams(name);
          this.defenceParams = params[0].defaults;
          this.defenceCategory = params[0].category;
          Object.entries(params[0].defaults).forEach(([key, entry]) => {
            this.types[key] = entry.default_value[0].constructor;
            this.info[key] = entry.info;
            this.updateDefenceParams(entry.pretty_name, entry.default_value, key);
          });
        }
        if (this.selectedAttack && this.selectedDefence) this.notReady = false;
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
        JSON.stringify(this.defenceParams),
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
    updateAttackParamsForm(name, newValues, paramKey, info) {
      const convertValues = this.convertForm(newValues, paramKey);
      this.attackParamsForm[paramKey] = { pretty_name: name, value: convertValues, info };
    },
    updateAttackParamsDropdown(name, newValues, newPossibleValues, paramKey, info) {
      const convertValues = this.convertDropdown(newValues, paramKey);
      const convertValuesTemp = this.convertDropdown(newPossibleValues, paramKey);
      this.attackParamsDropdown[paramKey] = {
        pretty_name: name,
        value: convertValues,
        values: convertValuesTemp,
        info,
      };
    },
    updateAttackParamsRange(name, newValues, paramKey, newMin, newMax, info) {
      const convertValues = this.convertRange(newValues, paramKey);
      this.attackParamsRange[paramKey] = {
        pretty_name: name,
        value: convertValues,
        minimum: newMin,
        maximum: newMax,
        info,
      };
    },
    updateDefenceParams(name, newValues, paramKey, info) {
      const convertValues = this.convertForm(newValues, paramKey);
      this.defenceParams[paramKey] = { pretty_name: name, value: convertValues, info };
    },
    convertForm(values, paramKey) {
      const type = this.typesForm[paramKey];
      return values.map(type);
    },
    convertDropdown(values, paramKey) {
      const type = this.typesDropdown[paramKey];
      return values.map(type);
    },
    convertRange(values, paramKey) {
      const type = this.typesRange[paramKey];
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
