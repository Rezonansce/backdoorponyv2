'''
For documentation check the README inside the defences/poisoning folder.
'''
from copy import deepcopy

import numpy as np
import torch
from tqdm import tqdm
from transformers import GPT2TokenizerFast, GPT2LMHeadModel


__name__ = 'onion'
__category__ = 'transformer'
__input_type__ = 'text'
__defaults__ = {
    'threshold': {
        'pretty_name': 'Threshold',
        'default_value': [20],
        'info': 'Suspicion score is determined by calculating the difference of perplexity between a sentence with(p0) and without(pi) every word - so suspicion = p0-pi. All words with suspicion score larger than threshold will be considered outliers and be removed before training the model.'
    }
}
__link__ = 'https://arxiv.org/pdf/2011.10369.pdf'
__info__ = '''ONION defence utilizes a pre-trained GPT-2 to find 
            and remove outliers(suspicious words) from the dataset
            before feeding it to the neural network'''.replace('\n', '')




def run(clean_classifier, test_data, execution_history, defence_params):
    '''Runs the onion defence

    Parameters
    ----------
    clean_classifier :
        Classifier that has not been tampered with, i.e. is clean
    test_data :
        Data that the clean classifier will be validated on as a tuple with (inputs, labels)
    execution_history :
        Dictionary with paths of attacks/defences taken to achieve classifiers, if any
    defence_params :
        Dictionary with the parameters for the defence (one value per parameter)

    Returns
    ----------
    Returns the updated execution history dictionary
    '''
    print('Instantiating ONION defence.')

    key_index = 0
    test_text = test_data[0]

    new_execution_history = deepcopy(execution_history)

    for entry in execution_history.values():
        for ts in defence_params['threshold']['value']:
            new_entry = deepcopy(entry)

            defence_classifier = ONION(deepcopy(clean_classifier), deepcopy(test_text), ts)

            def poison_condition(x): return x == -1

            new_entry.update({
                'defence': __name__,
                'defenceCategory': __category__,
                'threshold': ts,
                'dict_others': {
                    'poison_classifier': deepcopy(defence_classifier),
                    'poison_inputs': deepcopy(entry['dict_others']['poison_inputs']),
                    'poison_labels': deepcopy(entry['dict_others']['poison_labels']),
                    'poison_condition': deepcopy(poison_condition)
                }
            })

            key_index += 1
            new_execution_history.update(
                {'onion' + str(key_index): new_entry})

    return new_execution_history


class ONION:
    def __init__(self, classifier, clean_data, threshold):
        '''
        Apply the ONION defence to the classifier.

        Parameters
        ----------
        classifier :
            The classifier that has been poisoned
        x_clean_test :
            Clean (unpoisoned) preprocessed test data
        threshold:
            if suspicion score is above the give threshold,
            data is considered to be poisoned

        Returns
        ----------
        None
        '''
        self.keys = list(classifier.vocab.keys())
        self.classifier = classifier
        self.clean_data = clean_data
        self.threshold = threshold

        # GPT2 fast tokenizer for embedding
        self.gpt2_tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

        # GPT2 model with head layers to calculate perplexity
        self.gpt2_model = GPT2LMHeadModel.from_pretrained("gpt2")

        # move the model to gpu if possible
        self.gpt2_model.parallelize()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def calculate_suspicion(self, sentence):
        '''
            Calculate suspicion score of each word in the sentence

        Parameters
        ----------
        sentence:
            a string line consisting of multiple words

        Returns
        -------
            suspicion scores as a 1d numpy float array
        '''
        # tokenize
        tokenized_sentence = self.gpt2_tokenizer.encode(sentence, return_tensors="pt").to(self.device)
        # calculate perplexity p0 of the original sentence
        with torch.no_grad():
            original_perplexity = self.gpt2_model(tokenized_sentence, labels=tokenized_sentence)[0].detach().cpu().numpy()

        # split array to construct new sentences without a word
        arr_sentence = np.array(str.split(sentence))
        # initialize suspicion_scores array
        suspicion_scores = np.empty(len(arr_sentence), dtype=float)

        for i in range(len(arr_sentence)):
            # construct a new sentence without a word positioned at i
            sentence_with_removed_word = ' '.join(arr_sentence[np.arange(arr_sentence.size) != i])
            # tokenize
            new_tokenized_sentence = self.gpt2_tokenizer.encode(sentence_with_removed_word, return_tensors="pt").to(self.device)
            # calculate new perplexity pi
            with torch.no_grad():
                    new_perplexity = self.gpt2_model(new_tokenized_sentence, labels=new_tokenized_sentence)[0].detach().cpu().numpy()

            # suspicion score = p0 - pi
            sus_score = original_perplexity - new_perplexity

            suspicion_scores[i] = sus_score

        return suspicion_scores

    def predict_whether_poisoned(self, data):
        '''
        Calculates suspicion of all words in the sentence,
        uses the highest suspicion score when comparing to the user-selected threshold

        Parameters
        ----------
        data:
            dataset to classify as poisoned/not

        Returns
        -------
        A boolean array representing whether an entry is poisoned or not
        '''
        preds = []
        # i = 0
        for entry in tqdm(data):
            # print(i)
            # i+=1
            sentence = self.construct_sentence(entry)
            if len(sentence) > 1:
                suspicion_scores = self.calculate_suspicion(sentence)
                highest_suspicion = np.exp(np.max(suspicion_scores))
                preds.append(highest_suspicion)
                print(highest_suspicion)
            else:
                preds.append(0)
        return np.array(preds) > self.threshold

    def construct_sentence(self, sentence):
        '''
        Constructs a sentence from vocabulary given ids

        Parameters
        ----------
        sentence - ids of words in the vocabulary

        Returns
        -------
        sentence as a string

        '''
        constructed_sentence = []
        for i, word_index in enumerate(sentence):
            if word_index != 0:
                constructed_sentence.append(self.keys[word_index-1])
        return ' '.join(constructed_sentence)

    def predict(self, data):
        '''Classifies the input if unpoisoned,
        otherwise marks as poisoned and skips the entry

        Parameters
        ----------
        data:
            the dataset to classify

        Returns
        -------
        preds:
            1d numpy array of predictions with entry set to -1 if gp2 calculated suspicion score
            is higher than the user-selected threshold
        '''
        poison_conditions = self.predict_whether_poisoned(data)
        preds_classifier = self.classifier.predict(data)
        preds = np.empty(len(data), dtype=float)
        for i, pc in enumerate(poison_conditions):
            if pc:
                preds[i] = -1
            else:
                preds[i] = preds_classifier[i]

        return preds

    def get_predictions(self, x_poisoned_data):
        '''
        Verify the defence.

        Parameters
        ----------
        x_poisoned_test :
            The poisoned preprocessed test set to verify the defence effectiveness

        Returns
        ----------
        poison_preds :
            Suspicion score predictions for poisoned dataset
        clean_preds :
            Suspicion score predictions for clean dataset
        '''
        poison_preds = self.predict(x_poisoned_data)
        clean_preds = self.predict(self.clean_data)

        return poison_preds, clean_preds
