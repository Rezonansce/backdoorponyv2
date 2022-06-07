<template>
  <div >
    <v-dialog
      width="700"
      scrollable
      height=""
    >
      <template v-slot:activator="{ on, attrs }">
        <v-btn
          icon
          color="accent"
          v-bind="attrs"
          v-on="on"
        >
          <v-icon >
            mdi-information
          </v-icon>
        </v-btn>
      </template>
      <template v-slot:default="dialog">
      <v-card color="tertiary" height="300">
        <v-card-title class="text-h5">
          <v-icon color="accent" class="mx-4">
            mdi-information
          </v-icon>
          {{ actionName }}
        </v-card-title>

        <v-card-text>
          {{ infoText }}
        </v-card-text>
        <v-card-text v-if="hasLink">
          For more info check out this link:
          <a v-bind:href="link" target="_blank" style="color:#E94560">
              {{ actionName }}
          </a>
        </v-card-text>
        <v-divider></v-divider>

        <v-card-actions>
          <v-spacer></v-spacer>
          <v-btn
            color="white"
            text
            @click='dialog.value = false'
          >
            OK
          </v-btn>
        </v-card-actions>
      </v-card>
      </template>
    </v-dialog>
  </div>
</template>

<script>

export default ({
  setup() {

  },
  props: {
    actionName: String,
    link: String,
    infoText: String,
    hasLink: Boolean,
  },
});
</script>
