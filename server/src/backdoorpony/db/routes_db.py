from backdoorpony import app
from flask import jsonify, request

from extensions import db
from models import Action, Input, Param
from seeder import seed_with_predefined

db.init_app(app)


@app.route('/seed')
def seed():
    '''Seeds the database.'''
    with app.app_context():
        db.create_all()
    seed_with_predefined(db)
    return 'Database seeded!'


# Actions ---------------------------------------------------------------


@app.route('/get_all_actions_db')
def get_all_actions_db():
    '''Returns all the actions.'''
    jsonString = []
    actions = Action.query.all()
    for action in actions:
        jsonString.append(action.as_dict())
    return jsonify(jsonString)

# Params ---------------------------------------------------------------


@app.route('/get_all_params_db', methods=['GET'])
def get_all_params_db():
    '''Returns a list of all the parameters in JSON format.'''
    jsonString = []
    params = Param.query.all()
    for param in params:
        jsonString.append(param.as_dict())
    return jsonify(jsonString)

# Attacks ---------------------------------------------------------------


@app.route('/get_attack_db', methods=['GET'])
def get_attack_db():
    '''Returns an attack with a specific name in JSON format.'''
    # attack_name = request.form['attackName'] # Uncomment to use forms like in the /execute route
    attack_name = request.args.get('name', default='badnet', type=str)
    attack = Action.query.filter_by(is_attack=True, name=attack_name).first()
    return jsonify(attack.as_dict())


@app.route('/get_all_attacks_db', methods=['GET'])
def get_all_attacks_db():
    '''Returns a list of all the attacks in JSON format.'''
    jsonString = []
    attacks = Action.query.filter_by(is_attack=True).all()
    for attack in attacks:
        jsonString.append(attack.as_dict())
    return jsonify(jsonString)


# Defences ---------------------------------------------------------------


@app.route('/get_defence_db', methods=['GET'])
def get_defence_db():
    '''Returns a defence with a specific name in JSON format.'''
    # defence_name = request.form['defenceName'] # Uncomment to use forms like in the /execute route
    defence_name = request.args.get('name', default='strip', type=str)
    defence = Action.query.filter_by(
        is_attack=False, name=defence_name).first()
    return jsonify(defence.as_dict())


@app.route('/get_all_defences_db', methods=['GET'])
def get_all_defences_db():
    '''Returns a list of all the defences in JSON format.'''
    jsonString = []
    defences = Action.query.filter_by(is_attack=False).all()
    for defence in defences:
        jsonString.append(defence.as_dict())
    return jsonify(jsonString)

# Defaults ------------------------------------------------------------


@app.route('/get_params_action_db', methods=['GET'])
def get_params_action_db():
    '''Gets all suitable parameters for an action with a specific name.'''
    jsonString = []
    # action_name = request.form['attackName'] # Uncomment to use forms like in the /execute route
    action_name = request.args.get('name', default='badnet', type=str)
    defaults = Param.query.filter(Param.defaults.any(name=action_name)).all()

    for default in defaults:
        jsonString.append(default.as_dict())
    return jsonify(jsonString)

# Inputs---------------------------------------------------------------


@app.route('/get_input_types_action_db', methods=['GET'])
def get_input_types_action_db():
    '''Gets all suitable input types for an action with a specific name.'''
    jsonString = []
    # action_name = request.form['attackName'] # Uncomment to use forms like in the /execute route
    action_name = request.args.get('name', default='strip', type=str)
    configs = Input.query.filter(Input.configs.any(name=action_name)).all()

    for config in configs:
        jsonString.append(config.as_dict())
    return jsonify(jsonString)
