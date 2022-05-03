import axios from 'axios';

const apiConfig = {
  baseUrl: 'http://localhost:5000',
};

export default {
  get: (uri) => axios.get(`${apiConfig.baseUrl}${uri}`),
  post: (uri, data, config = {}) => axios.post(`${apiConfig.baseUrl}${uri}`, data, config),
  put: (uri, data, config = {}) => axios.put(`${apiConfig.baseUrl}${uri}`, data, config),
};
