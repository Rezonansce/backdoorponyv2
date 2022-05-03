import http from '../http-common';

class MetricsService {
  getResults = () => http.get('/results').then((resp) = resp.data);
}

export default new MetricsService;
