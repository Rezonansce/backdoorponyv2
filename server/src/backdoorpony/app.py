# These are the packages we are inspecting
import copy
import json


import backdoorpony.attacks
import backdoorpony.defences
import backdoorpony.metrics
from backdoorpony.app_tracker import AppTracker
from backdoorpony.dynamic_imports import import_submodules_attributes
from flask import Flask, jsonify, request
# Instantiate the app
from flask_cors import CORS, cross_origin

# temporary map
dataset_to_model = {
    "IMDB": "IMDB_LSTM_RNN",
    "MNIST": "MNIST_CNN",
    "CIFAR10": "CifarCNN",
    "Fashion_MNIST": "FMNIST_CNN",
    "Audio_MNIST": "Audio_MNIST_RNN"
    "AIDS": "AIDS_sage",
    "Mutagenicity": "Mutagenicity_sage",
    "IMDB MULTI": "IMDB_MULTI_sage",
    "Yeast": "Yeast_sage",
    "Synthie": "Synthie_sage"
}


app = Flask(__name__)
app.config.from_object(__name__)
app.config['SECRET_KEY'] = 'df0a17bc371e1b72883f3df3cc0928dd'
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+mysqlconnector://pony:backdoor@mariadb:3306/bpdb'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Enable CORS

CORS(app, resources={r'/*': {'origins': '*'}})

app_tracker = AppTracker()

@app.route('/home')
def home():
    '''Returns welcome string.'''
    jsonString = 'Welcome to BackdoorPony!'
    return jsonString


# Models & Classifiers -------------------------------------------------

@app.route('/get_datasets', methods=['GET'])
def get_datasets():
    '''Provides all available datasets

    Should return the following:
    {
        'image': [
            'MNIST',
            'CIFAR10'
        ],
        'text': [
            'IMDB'
        ],
        'audio': [...],
        'graph': [...]
    }
    '''
    return jsonify(app_tracker.model_loader.get_datasets())


@app.route('/select_model', methods=['POST'])
@cross_origin()
def select_model():
    '''Select which model is used to create the classifier.
    Can be either a built-in one or have a file with the model attached which is then used.


    Should receive the following, where values between <> can vary:
    form:
        {
            'type': '<image>',
            'dataset': '<MNIST>',
        }
    file (optional):
        {
            'model': <.pth file>
        }
    '''
    model_params = json.loads(request.form['modelParams'].replace("'", '"'))
    app_tracker.dataset = request.form['dataset']
    model = None
    if 'model' in request.files:
        model = request.files['model']
        app_tracker.file_name = model.filename
    app_tracker.model_loader.make_classifier(request.form['type'],
                                 request.form['dataset'],
                                 model_params,
                                 model)

    return jsonify('Creating/choosing the classifier was successful.')


# Attacks & Defences ---------------------------------------------------
@app.route('/get_all_models', methods=['GET'])
def get_all_models():
    '''Returns a list of all the attacks and their info in JSON format.'''
    _, models = import_submodules_attributes(package=backdoorpony.models, result=[
    ], recursive=True, req_module=None, req_attr=['__name__', '__category__','__input_type__', '__info__', '__link__'])
    return jsonify(models)

@app.route('/get_all_attacks', methods=['GET'])
def get_all_attacks():
    '''Returns a list of all the attacks and their info in JSON format.'''
    _, attacks = import_submodules_attributes(package=backdoorpony.attacks, result=[
    ], recursive=True, req_module=None, req_attr=['__name__', '__category__','__input_type__', '__info__', '__link__'])
    return jsonify(attacks)


@app.route('/get_all_defences', methods=['GET'])
def get_all_defences():
    '''Returns a list of all the defences and their info in JSON format.'''
    _, defences = import_submodules_attributes(package=backdoorpony.defences, result=[], recursive=True, req_module=None, req_attr = ['__name__', '__category__', '__input_type__', '__info__', '__link__'])
    return jsonify(defences)


@app.route('/get_stored_attack_name', methods=['GET'])
def get_stored_attack_name():
    '''Returns the stored attack name.'''
    return app_tracker.attack_name


@app.route('/get_stored_defence_name', methods=['GET'])
def get_stored_defence_name():
    '''Returns the stored defence name.'''
    return app_tracker.defence_name


@app.route('/get_stored_attack_category', methods=['GET'])
def get_stored_attack_category():
    '''Returns the stored attack category.'''
    return app_tracker.attack_category


@app.route('/get_stored_defence_category', methods=['GET'])
def get_stored_defence_category():
    '''Returns the stored defence category.'''
    return app_tracker.defence_category


# Params ---------------------------------------------------------------

@app.route('/get_default_model_params', methods=['POST'])
def get_default_model_params():
    '''Returns a list of all the default model parameters in JSON format.'''
    dataset_name = request.form['modelName']
    model_name = dataset_to_model[dataset_name]
    _, default_params = import_submodules_attributes(package=backdoorpony.models, result=[
    ], recursive=True, req_module=model_name, req_attr=['__category__', '__defaults__'], debug=False)
    return jsonify(default_params)

@app.route('/get_default_attack_params', methods=['POST'])
def get_default_attack_params():
    '''Returns a list of all the default attack parameters in JSON format.'''
    attack_name = request.form['attackName'].lower()
    _, default_params = import_submodules_attributes(package=backdoorpony.attacks, result=[
    ], recursive=True, req_module=attack_name, req_attr=['__category__', '__defaults__'])
    return jsonify(default_params)


@app.route('/get_default_defence_params', methods=['POST'])
def get_default_defence_params():
    '''Returns a list of all the default defence parameters in JSON format.'''
    defence_name = request.form['defenceName'].lower()
    _, default_params = import_submodules_attributes(package=backdoorpony.defences, result=[], recursive=True, req_module=defence_name, req_attr = ['__category__', '__defaults__'])
    return jsonify(default_params)


@app.route('/get_stored_attack_params', methods=['GET'])
def get_stored_attack_params():
    '''Returns the dictionary storing attack parameters in JSON format.'''
    return jsonify(app_tracker.attack_params)


@app.route('/get_stored_defence_params', methods=['GET'])
def get_stored_defence_params():
    '''Returns the dictionary storing defence parameters in JSON format.'''
    return jsonify(app_tracker.defence_params)


# Execute ------------------------------------------------------------

@app.route('/execute', methods=['POST'])
@cross_origin()
def execute():
    '''Executes the selected attack and/or defence with their corresponding parameters
    If attackName is not in the form, no attack will be executed.
    If defenceName is not in the form, no defence will be executed.

    in form:
        attackName
        defenceName
        attackParams
        defenceParams
        attackCategory
        defenceCategory
    '''
    app_tracker.reset_action_info()
    clean_classifier = app_tracker.model_loader.get_classifier()
    test_data = app_tracker.model_loader.get_test_data()
    if hasattr(app_tracker.model_loader, 'audio'):
        test_data = app_tracker.model_loader.audio_test_data
    execution_history = {}

    if 'attackName' in request.form:
        app_tracker.attack_name = request.form['attackName']
        app_tracker.attack_category = request.form['attackCategory']
        app_tracker.attack_params = json.loads(request.form['attackParams'].replace("'", '"'))
        train_data = app_tracker.model_loader.get_train_data()
        if hasattr(app_tracker.model_loader, 'audio'):
            train_data = app_tracker.model_loader.audio_train_data
        execution_history = app_tracker.action_runner.run_attack(clean_classifier=clean_classifier,
                                                                 train_data=train_data,
                                                                 test_data=test_data,
                                                                 execution_history=execution_history,
                                                                 attack_to_run=app_tracker.attack_name,
                                                                 attack_params=app_tracker.attack_params)

    if hasattr(app_tracker.model_loader, 'audio'):
        test_data = app_tracker.model_loader.get_test_data()
    if 'defenceName' in request.form:
        app_tracker.defence_name = request.form['defenceName']
        app_tracker.defence_category = request.form['defenceCategory']
        app_tracker.defence_params = json.loads(request.form['defenceParams'].replace("'", '"'))
        execution_history = app_tracker.action_runner.run_defence(clean_classifier=clean_classifier,
                                                                  test_data=test_data,
                                                                  execution_history=execution_history,
                                                                  defence_to_run=app_tracker.defence_name,
                                                                  defence_params=app_tracker.defence_params)


    app_tracker.main_metrics_runner.instantiate(clean_classifier=clean_classifier,
                                                execution_history=execution_history,
                                                benign_inputs=test_data,
                                                requests={})

    return jsonify('Execution of attack and/or defence was successful.')


# Metrics -------------------------------------------------------------

@app.route('/get_metrics_info', methods=['GET'])
@cross_origin()
def get_metrics_info():
    '''Gets metrics available for the category of attack and or defence saved in the app_tracker
    Returns a dictionary with the following shape:
    {
        #for attack only
        <name1>: {
            name: <name of the metric>,
            pretty_name: <human-readable name of metric>,
            info: <information on the metric>,
            is_attack_metric: True
        },
        #for defence only
        <name2>: {
            name: <name of the metric>,
            pretty_name: <human-readable name of metric>,
            info: <information on the metric>,
            is_defence_metric: True
        },
        #for both attack and defence
        <name3>: {
            name: <name of the metric>,
            pretty_name: <human-readable name of metric>,
            info: <information on the metric>,
            is_attack_metric: True/False,
            is_defence_metric: True/False
        }
    }
    '''
    _, metrics_info = import_submodules_attributes(package=backdoorpony.metrics, result=[], recursive=True, req_module=None, req_attr = ['__category__', '__info__'])

    available_metrics = {}

    if app_tracker.attack_category:
        for info in metrics_info:
            if info['category'] == app_tracker.attack_category:
                attack_metrics_info = info
                break
        for key, val in attack_metrics_info['info'].items():
            new_val = available_metrics.get(key, copy.deepcopy(val))
            new_val.update({'is_attack_metric': True, 'name': key})
            available_metrics.update({key: new_val})

    if app_tracker.defence_category:
        for info in metrics_info:
            if info['category'] == app_tracker.defence_category:
                defence_metrics_info = info
                break
        for key, val in defence_metrics_info['info'].items():
            new_val = available_metrics.get(key, copy.deepcopy(val))
            new_val.update({'is_defence_metric': True, 'name': key})
            available_metrics.update({key: new_val})

    return jsonify(available_metrics)


@app.route('/get_metrics_results', methods=['POST'])
def get_metrics_results():
    '''Calculates the metrics in the request and returns in graph-compatible format
    '''
    request_metrics = json.loads(request.form['request'].replace("'", '"'))

    app_tracker.main_metrics_runner.update(requests=request_metrics)

    return jsonify(app_tracker.main_metrics_runner.get_results())


# Configuration ----------------------------------------------------------

@app.route('/get_configuration_file', methods=['GET'])
def get_configuration_file():
    '''Sends a file with the details of the executed configuration.
    Returns a dictionary containing values used for attack and/or defence name, category, input_type, parameters.
    '''
    return jsonify(app_tracker.generate_configuration_file())




