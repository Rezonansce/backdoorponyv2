import http from '../http-common';

class AttacksAndDefencesService {
  response = {}

  getResponse = () => this.response;

  setResponse = (response) => {
    this.response = response;
  }

  getAttacks = () => http.get('/get_all_attacks').then((resp) => resp.data);

  getDefences = () => http.get('/get_all_defences').then((resp) => resp.data);

  getAttackParams = (name) => {
    const formData = new FormData();
    formData.append('attackName', name);
    return http.post('/get_default_attack_params', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    }).then((response) => response.data);
  }

  getDefenceParams = (name) => {
    const formData = new FormData();
    formData.append('defenceName', name);
    return http.post('/get_default_defence_params', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    }).then((response) => response.data);
  }

  execute = (isDefended,
    isAttacked,
    attackName,
    attackParams,
    defenceName,
    defenceParams,
    attackCategory,
    defenceCategory) => {
    const formData = new FormData();
    if (isAttacked) {
      formData.append('attackName', attackName.replace(/^"(.*)"$/, '$1'));
      formData.append('attackParams', attackParams);
      formData.append('attackCategory', attackCategory.replace(/^"(.*)"$/, '$1'));
    }
    if (isDefended) {
      formData.append('defenceName', defenceName.replace(/^"(.*)"$/, '$1'));
      formData.append('defenceParams', defenceParams);
      formData.append('defenceCategory', defenceCategory.replace(/^"(.*)"$/, '$1'));
    }
    return http.post('/execute', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    }).then((response) => response.data)
      .catch((error) => console.log(error.response));
  }
}

export default new AttacksAndDefencesService();
