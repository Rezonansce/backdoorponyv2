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

  // execute = (isDefended,
  //   isAttacked,
  //   attackName,
  //   attackParams,
  //   defenceName,
  //   defenceParams,
  //   attackCategory,
  //   defenceCategory) => {
  //   const formData = new FormData();
  //   if (isAttacked) {
  //     formData.append('attackName', attackName.replace(/^"(.*)"$/, '$1'));
  //     formData.append('attackParams', attackParams);
  //     formData.append('attackCategory', attackCategory.replace(/^"(.*)"$/, '$1'));
  //   }
  //   if (isDefended) {
  //     formData.append('defenceName', defenceName.replace(/^"(.*)"$/, '$1'));
  //     formData.append('defenceParams', defenceParams);
  //     formData.append('defenceCategory', defenceCategory.replace(/^"(.*)"$/, '$1'));
  //   }
  //   return http.post('/execute', formData, {
  //     headers: {
  //       'Content-Type': 'multipart/form-data',
  //     },
  //   }).then((response) => response.data)
  //     .catch((error) => console.log(error.response));
  // }
}

export default new ModelService();
