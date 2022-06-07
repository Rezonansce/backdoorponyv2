import http from '../http-common';

class PlotServices {
  response = {};

  setResponse =(response) => {
    this.response = response;
  }

  getResponse = () => this.response;

  getAttackCategory = () => http.get('/get_stored_attack_category').then((resp) => resp.data);

  getDefenceCategory = () => http.get('/get_stored_defence_category').then((resp) => resp.data);

  getAttackParams = () => http.get('/get_stored_attack_params').then((resp) => resp.data);

  getDefenceParams = () => http.get('/get_stored_defence_params').then((resp) => resp.data);

  getMetrics = () => http.get('/get_metrics_info').then((resp) => resp.data);

  getConfigurationFile = () => http.get('/get_configuration_file').then((resp) => resp.data);

  plot = (plots) => {
    const formData = new FormData();
    formData.append('request', plots);
    return http.post('/get_metrics_results', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    }).then((response) => response.data);
  }
}

export default new PlotServices();
