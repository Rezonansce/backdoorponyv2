from backdoorpony.metrics.main_metrics_runner import MainMetricsRunner
from backdoorpony.models.loader import Loader
from backdoorpony.runners.runner import Runner


class AppTracker():
    '''A class used to track the application's state.

    It is instantiated when the application starts.
    It stores the necessary attributes to return to the front end throughout multiple screens. 
    The attributes are reset when the execute route function is called.
    '''
    def __init__(self):
        """
        Fields
        ----------
        model_loader : Loader
            The the loader which creates a classifier
        main_metrics_runner : MainMetricsRunner
            The metrics runner used for calculating metrics results
        action_runner : Runner
            The action runner used for executing attacks/defences.
        attack_name : str
            The name of the currently stored attack.
        defence_name : str
            The name of the currently stored defence.
        attack_category : str
            The category of the currently stored attack.    
        defence_category : str
            The category of the currently stored defence.
        attack_params : str
            The parameters of the currently stored attack.    
        defence_params : str
            The parameters of the currently stored defence.    

        Returns
        ----------
        None
        """
        self.dataset = ''
        self.file_name = ''
        self.model_loader = Loader()
        self.main_metrics_runner = MainMetricsRunner()
        self.action_runner = Runner()
        self.attack_name = ''
        self.defence_name = ''
        self.attack_category = ''
        self.defence_category = ''
        self.attack_params_form = {}
        self.attack_params_dropdown = {}
        self.attack_params_slidebar = {}
        self.attack_params_combined = {}
        self.defence_params_form = {}
        self.defence_params_dropdown = {}
        self.defence_params_slidebar = {}
        self.defence_params_combined = {}
        self.executing = False
        self.execution_thread = None

    def reset_action_info(self):
        '''Resets the fields on attacks and defences

        Fields are:
            attack_name
            attack_category
            attack_params
            defence_name
            defence_category
            defence_params

        Returns
        ----------
        None
        '''
        self.attack_name = ''
        self.attack_category = ''
        self.attack_params_form = {}
        self.attack_params_dropdown = {}
        self.attack_params_slidebar = {}
        self.attac_params_combined = {}
        self.defence_name = ''
        self.defence_category = ''
        self.defence_params = {}

    def generate_configuration_file(self):
        '''Generates step-by-step instructions for recreating an attack/defence configuration

        Uses the last used values and is represented by a list of strings.
        E.g.:       
        [
            "1. Select the MNIST dataset;",
            "2. Upload your initial model named 'cmnist.model.pth';",
            "3. Choose the badnet attack;",
            "    3a. Input [0.33] for percentage of poison;",
            "    3b. Input [2] for target class;",
            "    3c. Input ['pattern'] for style of trigger;",
            "4. Select 'no defence';",
            "5. Press execute."
        ]

        Returns
        ----------
        List of strings with instructions
        '''
        step = 1
        ls = ['{0}. Select the {1} dataset;'.format(step, self.dataset)]
        step +=1
        if self.file_name:
            ls.append('{0}. Upload your initial model named \'{1}\';'.format(step, self.file_name))
        else:
            ls.append('{0}. Select \'use built-in model\';'.format(step))
        step +=1
        if self.attack_name:
            ls.append('{0}. Choose the {1} attack;'.format(step, self.attack_name))
            alph_count = 97
            for param in self.attack_params_form.values():
                ls.append('    {0}{1}. Input {2} for {3};'.format(step,
                                                             chr(alph_count),
                                                             str(param['value']),
                                                             param['pretty_name'].lower()))
                alph_count +=1
        else:
            ls.append('{0}. Select \'no attack\';'.format(step))
        step +=1
        if self.defence_name:
            ls.append('{0}. Choose the {1} defence;'.format(step, self.defence_name))
            alph_count = 97
            for param in self.defence_params.values():
                ls.append('    {0}{1}. Input {2} for {3};'.format(step,
                                                             chr(alph_count),
                                                             str(param['value']),
                                                             param['pretty_name'].lower()))
                alph_count +=1
        else:
            ls.append('{0}. Select \'no defence\';'.format(step))
        step +=1
        ls.append('{0}. Press execute.'.format(step))
        return ls
    
