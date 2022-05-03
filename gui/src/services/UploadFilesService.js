import http from '../http-common';

class UploadFilesService {
  getDataSets = () => http.get('/get_datasets').then((resp) => resp.data);

  upload = (type, dataset, model) => {
    const formData = new FormData();
    formData.append('type', type);
    formData.append('dataset', dataset);
    formData.append('model', model);
    return http.post('/select_model', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
        'Access-Control-Allow-Origin': '*',
      },
    });
  }
}

export default new UploadFilesService();
