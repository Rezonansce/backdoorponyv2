import os
import tempfile

import pytest
import json
from flask import jsonify
from unittest.mock import patch

import flask
from backdoorpony.app import app


# Creates the test client as it is done in the Flask documentation
@pytest.fixture
def client():
    db_fd, flask.app.config['DATABASE'] = tempfile.mkstemp()
    flask.app.config['TESTING'] = True

    with flask.app.test_client() as client:
        with flask.app.app_context():
            flask.init_db()
        yield client

    os.close(db_fd)
    os.unlink(flask.app.config['DATABASE'])

# Tests the route for getting all attacks
with app.test_client() as c:
    with patch('backdoorpony.app.import_submodules_attributes', return_value=(None, '''[
{
    'info': 'Badnet is an attack that adds a backdoor to a neural network by retraining the neural network on partially poisoned input. The input is poisoned by adding a visual trigger to it. This trigger could be a pattern or just a single pixel.', 
    'link': 'https://arxiv.org/pdf/1708.06733.pdf', 
    'name': 'badnet'
  }
]''')):
        # The answer is in json format, so get_json() is used.
        # This is a get request, c.post('route') can also be done.
        rv = c.get('/get_all_attacks')
        json_data = rv.get_json()

        ans = '''[
{
    'info': 'Badnet is an attack that adds a backdoor to a neural network by retraining the neural network on partially poisoned input. The input is poisoned by adding a visual trigger to it. This trigger could be a pattern or just a single pixel.', 
    'link': 'https://arxiv.org/pdf/1708.06733.pdf', 
    'name': 'badnet'
  }
]'''

        assert ans == str(json_data)

# Tests the route for getting all defences
with app.test_client() as c:
    with patch('backdoorpony.app.import_submodules_attributes', return_value=(None, '''[
  {
    'info': 'STRIP, or STRong Intentional Perturbation, is a run-time based trojan attack detection system that focuses on vision system. STRIP intentionally perturbs the incoming input, for instance, by superimposing various image patterns and observing the randomness of predicted classes for perturbed inputs from a given deployed model\u2014malicious or benign. A low entropy in predicted classes violates the input-dependence property of a benign model. It implies the presence of a malicious input\u2014a characteristic of a trojaned input.', 
    'link': 'https://arxiv.org/pdf/1902.06531.pdf', 
    'name': 'strip'
  }
]''')):
        rv = c.get('/get_all_defences')
        json_data = rv.get_json()

        ans = '''[
  {
    'info': 'STRIP, or STRong Intentional Perturbation, is a run-time based trojan attack detection system that focuses on vision system. STRIP intentionally perturbs the incoming input, for instance, by superimposing various image patterns and observing the randomness of predicted classes for perturbed inputs from a given deployed model\u2014malicious or benign. A low entropy in predicted classes violates the input-dependence property of a benign model. It implies the presence of a malicious input\u2014a characteristic of a trojaned input.', 
    'link': 'https://arxiv.org/pdf/1902.06531.pdf', 
    'name': 'strip'
  }
]'''

        assert ans == str(json_data)

# Tests the metrics info route
# with app.test_client() as c:
#     with patch('backdoorpony.app.import_submodules_attributes', return_value=(None, '''[
#   {
#     'category': 'poisoning', 
#     'info': {
#       'acc': {
#         'info': 'Calculated accuracy on benign input', 
#         'pretty_name': 'Accuracy'
#       }, 
#       'asr': {
#         'info': 'Calculated accuracy on poisoned input', 
#         'pretty_name': 'ASR'
#       }, 
#       'cad': {
#         'info': 'Difference between accuracy of the clean classifier and this classifier on benign input', 
#         'pretty_name': 'CAD'
#       }
#     }
#   }, 
#   {
#     'category': 'transformer', 
#     'info': {
#       'acc': {
#         'info': 'Calculated accuracy on benign input', 
#         'pretty_name': 'Accuracy'
#       }, 
#       'asr': {
#         'info': 'Calculated accuracy on poisoned input', 
#         'pretty_name': 'ASR'
#       }, 
#       'cad': {
#         'info': 'Difference between accuracy of the clean classifier and this classifier on benign input', 
#         'pretty_name': 'CAD'
#       }, 
#       'fp': {
#         'info': 'Probability that the classifier abstains on benign input (%)', 
#         'pretty_name': 'FP'
#       }, 
#       'tp': {
#         'info': 'Probability that the classifier abstains on poisoned input (%)', 
#         'pretty_name': 'TP'
#       }
#     }
#   }
# ]''')):
#         rv = c.get('/get_metrics_info')
#         json_data = rv.get_json()

#         ans = '''[
#   {
#     'category': 'poisoning', 
#     'info': {
#       'acc': {
#         'info': 'Calculated accuracy on benign input', 
#         'pretty_name': 'Accuracy'
#       }, 
#       'asr': {
#         'info': 'Calculated accuracy on poisoned input', 
#         'pretty_name': 'ASR'
#       }, 
#       'cad': {
#         'info': 'Difference between accuracy of the clean classifier and this classifier on benign input', 
#         'pretty_name': 'CAD'
#       }
#     }
#   }, 
#   {
#     'category': 'transformer', 
#     'info': {
#       'acc': {
#         'info': 'Calculated accuracy on benign input', 
#         'pretty_name': 'Accuracy'
#       }, 
#       'asr': {
#         'info': 'Calculated accuracy on poisoned input', 
#         'pretty_name': 'ASR'
#       }, 
#       'cad': {
#         'info': 'Difference between accuracy of the clean classifier and this classifier on benign input', 
#         'pretty_name': 'CAD'
#       }, 
#       'fp': {
#         'info': 'Probability that the classifier abstains on benign input (%)', 
#         'pretty_name': 'FP'
#       }, 
#       'tp': {
#         'info': 'Probability that the classifier abstains on poisoned input (%)', 
#         'pretty_name': 'TP'
#       }
#     }
#   }
# ]'''

#         assert ans == str(json_data)

with app.test_client() as c:
    rv = c.get('/home')

    ans = b'Welcome to BackdoorPony!'

    assert ans in rv.data
