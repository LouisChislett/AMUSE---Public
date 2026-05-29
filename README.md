# AMUSE: Adaptive Model Updating Using a Simulated Environment
Code for AMUSE

## Code

All code was run in Python 3.10.1

Please view the requirements.txt to see the required packages and versions used in the scripts.

For storage space reasons, simulated datasets and results have been left out of this repository, but they can be easily recreated by running the python scripts in the correct order.

Generally to recreate the synthetic results, run scripts for the associated experiment in the following order:
* DatasetCreation.py - Creates datasets (simulated real-world datasets)
* ModelUpdatingEnv.py - Defines simulated environment
* ReinforcementLearning.py - Learns initial policies for differnt hyperparameters
* HyperparameterTuning.py - Picks best hyperparameter
* RealEnvironment.py - Defines MDP environment for real world data
* Results.py - Benchmarks AMUSE against DDM, FRD and STEPD
* Scatterplots.py - Creates plots


Workflows is slightly differnt for ELEC2 and SPARRA datasets

SPARRA results can not be replicated without access to GS data