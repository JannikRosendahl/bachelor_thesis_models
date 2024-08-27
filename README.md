### Documentation for important files referenced in thesis.

Every notebook documents one experiment. Its artifacts (model, logs, plots) are stored in `./saves/$experiment_name`. The root directory contains the base versions of the experiments. Variants of the experiments are stored in subdirectories.


| File  | Description |
| --- | --- |
| [random_baseline_equalweights.ipynb](https://github.com/JannikRosendahl/bachelor_thesis_models/blob/master/random_baseline_equalweights.ipynb)<br>[random_baseline_suppweights.ipynb](https://github.com/JannikRosendahl/bachelor_thesis_models/blob/master/random_baseline_suppweights.ipynb) | random baseline with equal weighting for classes and support weighting for classes |
| [baseline_rnn.ipynb](https://github.com/JannikRosendahl/bachelor_thesis_models/blob/master/baseline_rnn.ipynb) | baseline model on which other models then try to improve |
| [feature_user.ipynb](https://github.com/JannikRosendahl/bachelor_thesis_models/blob/master/feature_user.ipynb) | baseline + user.<br>Variants with different model size, lstm layers, dropout in [here](https://github.com/JannikRosendahl/bachelor_thesis_models/tree/master/feature_user_variants). |
| [feature_objtype.ipynb](https://github.com/JannikRosendahl/bachelor_thesis_models/blob/master/feature_objtype.ipynb) | baseline + predicate object type.<br>Variants with different model size, lstm layers, dropout in [here](https://github.com/JannikRosendahl/bachelor_thesis_models/tree/master/feature_objtype_variants). |
| [feature_tlds.ipynb](https://github.com/JannikRosendahl/bachelor_thesis_models/blob/master/feature_tlds.ipynb) | baseline + predicate object top level directory path.<br>Variants with different architectures and diffent path-encodings in [here](https://github.com/JannikRosendahl/bachelor_thesis_models/tree/master/feature_tlds_variants). |
| [netinfo_output.ipynb](https://github.com/JannikRosendahl/bachelor_thesis_models/blob/master/netinfo_output.ipynb) | baseline + network information.<br>Variants with different model size, lstm layers in [here](https://github.com/JannikRosendahl/bachelor_thesis_models/tree/master/feature_netinfo_variants) |
| [deltatime_output.ipynb](https://github.com/JannikRosendahl/bachelor_thesis_models/blob/master/deltatime_output.ipynb) | baseline + time between events.<br>Variants with different architecture in [here]() |
| wip: [path_autoencoded.ipynb]() | todo |
| wip: [pioneer.ipynb]() | best model combining multiple features.<br>Variants in [here]() |
| | |
| [path_autoencoder.ipynb](https://github.com/JannikRosendahl/bachelor_thesis_models/blob/master/path_autoencoder.ipynb) | path autoencoder. saves |
| [util.py](https://github.com/JannikRosendahl/bachelor_thesis_models/blob/master/utils.py) | utility functions / shared code for experiments |
| [compare_report.ipynb](https://github.com/JannikRosendahl/bachelor_thesis_models/blob/master/compare_report.ipynb) | utility, uses exported classification reports of experiements to calculate metric differences for experiments |
