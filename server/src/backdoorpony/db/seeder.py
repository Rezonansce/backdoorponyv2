from pathlib import Path

import mysql.connector
from backdoorpony.models import Action, Input, Param


class DBManager:
    def __init__(self, database='bpdb', host='mariadb', user='pony', password='backdoor'):
        '''Initializes the database manager.'''
        self.connection = mysql.connector.connect(
            user=user,
            password=password,
            host=host,
            database=database
        )
        self.cursor = self.connection.cursor()

    def populate_db(self):
        '''Populates the database with given entries or tables.'''    

        # Retrieve path to /server/data
        path_tables = Path(__file__).parent / 'data/tables.sql'
        path_inserts = Path(__file__).parent / 'data/inserts.sql'

        # Read and execute SQL commands from /server/data/tables
        self.commit_from_file(path_tables)
        
        # Read and execute SQL commands from /server/data/inserts
        self.commit_from_file(path_inserts)

    def query_titles(self, table_name='actions'):
        '''Queries titles of certain tables and returns the result.
        Args:
            table_name (str): The name of the table.

        Returns:
            rec: The return value. A list containing the result of retrieving the titles from a table.
        '''
        self.cursor.execute('SELECT * FROM ' + table_name)
        rec = []
        for c in self.cursor:
            rec.append(c[0])
        return rec

    def query_all(self, table_name='actions', number_of_columns=4):
        '''Queries all information from certain tables and returns the result.
        Args:
            table_name (str): The name of the table.
            number_of_columns (int): The number of attributes a table has.

        Returns:
            rec: The return value. A list containing the result of retrieving all info from a table.
        '''
        self.cursor.execute('SELECT * FROM ' + table_name)
        rec = []
        for c in self.cursor:
            for i in range(number_of_columns):
                rec.append(c[i])
        return rec

    def commit_from_file(self, path_name):
        '''Executes and commits SQL statements from a file at a given path.'''
        with open(path_name) as file:
            commands = file.read()
            queries = commands.split(';')
            for query in queries:
                self.cursor.execute(query)
        self.connection.commit()
    
def seed_with_predefined(db):
    '''Seeds the database.'''
 
    # Adds possible actions
    actions = []

    strip = Action(name='STRIP', info='Compares clean input with the model.',
                link='https://arxiv.org/pdf/1909.02742.pdf', is_attack=False)

    badnet = Action(
        name='BadNet', info='Retrains on poisoned dataset.', link='https://arxiv.org/pdf/1708.06733.pdf', is_attack=True)

    trojanNN = Action(
        name='TrojanNN', info='Info on TrojanNN...', link='Link to TrojanNN', is_attack=True)
    
    actions.append(strip)
    actions.append(badnet)
    actions.append(trojanNN)

    for action in actions:
        db.session.add(action)

    # Adds possible parameters

    params = []

    # Generic params
    param_num_epochs = Param(name='Number of epochs',
                            info='The number of runs on the input.', value='10')

    # Attack params
    param_trigger_style = Param(name='Trigger style', info='The type of trigger of the attack.', value='pattern')
    param_poison_percentage = Param(name='Poison percentage', info='The percentage of data to be poisoned.', value='25')
    param_target_class = Param(name='Target class', info='The target class for the poisoned data to be misclassified as.', value='2')

    params.append(param_num_epochs)
    params.append(param_trigger_style)
    params.append(param_poison_percentage)
    params.append(param_target_class)

    # Defence params
    param_number_of_images = Param(name='Number of images', info='The number of images to be used.', value='100')

    params.append(param_number_of_images)
    
    for param in params:
        db.session.add(param)

    # Adds possible inputs

    input_image = Input(input_type='Image type')

    # Adds possible defaults of actions and params

    # Attack defaults
    badnet.defaults.append(param_num_epochs)
    badnet.defaults.append(param_trigger_style)
    badnet.defaults.append(param_poison_percentage)
    badnet.defaults.append(param_target_class)

    # Defence defaults
    strip.defaults.append(param_num_epochs)
    strip.defaults.append(param_number_of_images)

    # Adds possible configs of actions and input types
    
    # Attack configs
    badnet.configs.append(input_image)

    # Defence configs
    strip.configs.append(input_image)

    db.session.commit()
