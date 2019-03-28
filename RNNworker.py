import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH

from hpbandster.core.worker import Worker

import logging
logging.basicConfig(level=logging.DEBUG)

import tensorflow as tf
import implementation as imp
from runner import train_auto

class RnnWorker(Worker):
    def __init__(self, run_id, **kwargs):
        super().__init__(run_id, **kwargs)

    def compute(self, config, budget, *args, **kwargs):
        info = train_auto(config, budget)
        return ({
            "loss": 1-info["validation"][0],
            "info": {
                'train accuracy' : info["train"][0],
                'dev accuracy': info["validation"][0]
            }
        })

    @staticmethod
    def get_configspace():
            """
            It builds the configuration space with the needed hyperparameters.
            It is easily possible to implement different types of hyperparameters.
            Beside float-hyperparameters on a log scale, it is also able to handle categorical input parameter.
            :return: ConfigurationsSpace-Object
            """
            '''
            config = {
                'batch_size': [64, 128, 256, 512],
                'max_words': [137 - 250],
                'embedding': [50 - 100],
                'lr': [0.01-0.0001],
                'lstm_unit': [64, 128, 256, 512],
                'output_dim': [1 - 10],
                'layers': [1, 2, 3],
            }
            '''
            cs = CS.ConfigurationSpace()

            lr = CSH.UniformFloatHyperparameter('lr', lower=1e-5, upper=1e-2, default_value='1e-3', log=True)

            # embedding = CSH.UniformIntegerHyperparameter('embedding', lower=50, upper=100, default_value=50, log=True)
            max_words = CSH.UniformIntegerHyperparameter('max_words', lower=137, upper=250, default_value=200, log=True)
            output_dim = CSH.UniformIntegerHyperparameter('output_dim', lower=1, upper=10, default_value=3, log=True)

            # For demonstration purposes, we add different optimizers as categorical hyperparameters.
            # To show how to use conditional hyperparameters with ConfigSpace, we'll add the optimizers 'Adam' and 'SGD'.
            # SGD has a different parameter 'momentum'.
            batch_size = CSH.CategoricalHyperparameter('batch_size', [64, 128, 256, 512])
            lstm_unit = CSH.CategoricalHyperparameter('lstm_unit', [64, 128, 256, 512])
            layers = CSH.CategoricalHyperparameter('layers', [1, 2, 3])

            cs.add_hyperparameters([lr, max_words, output_dim, batch_size, lstm_unit, layers])

            return cs

if __name__ == "__main__":
    worker = RnnWorker(run_id='0')
    cs = worker.get_configspace()

    config = {
        'batch_size': 128,
        'max_words': 150,
        'lr': 0.001,
        'lstm_unit': 64,
        'output_dim': 3,
        'layers': 3,
    }

#     print(config)
    res = worker.compute(config=config, budget=100)
    print(res["info"])