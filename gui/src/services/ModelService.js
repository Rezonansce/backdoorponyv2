import http from '../http-common';

class ModelService {
  response = {}

  getResponse = () => this.response;

  setResponse = (response) => {
    this.response = response;
  }

  getModels = () => http.get('/get_all_models').then((resp) => resp.data);

  getModelParams = (name) => {
    const formData = new FormData();
    formData.append('modelName', name);
    return http.post('/get_default_model_params', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    }).then((response) => response.data);
  }

  selectModel = (
    type,
    dataset,
    modelParams,
  ) => {
    const formData = new FormData();
    formData.append('type', type);
    formData.append('dataset', dataset);
    formData.append('modelParams', modelParams);
    return http.post('/select_model', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
        'Access-Control-Allow-Origin': '*',
      },
    });
  }
}

export default new ModelService();
