# MoonBoardRNN

This repository features 3 key models:

(1) BetaMove, a preprocessing code which converts a MoonBoard problem into a move sequence that is similar to the predicted move sequence of climbing experts.

(2) DeepRouteSet, which was trained using the sequence data generated by BetaMove. Similar to music generation, DeepRouteSet learns the pattern between adjacent moves from existing MoonBoard problems, and is able to generate new problems.

(3) GradeNet, which was also trained using the sequence data generated by BetaMove, and is able to predict the Grade of a MoonBoard problem.

This is the project repository for the CS230 Spring 2020 course project, and is jointly developed by Yi-Shiou Duh and Je-Rui Chang. All our experiments were done in jupyter notebook format.

## Overview of our repository

### `scraping`
The scraping code is cloned and modified from https://github.com/gestalt-howard/moonGen.

### `raw_data`
This folder contains the hold difficulty scores evaluated by climbing experts (`\raw_data\HoldFeature2016LeftHand.csv`, `\raw_data\HoldFeature2016RightHand.csv`, `\raw_data\HoldFeature2016.xlsx`) and our scraped raw data from the MoonBoard website (`\raw_data\moonGen_scrape_2016_final`).

### `preprocessing`
There are 4 jupyter notebooks in the `preprocessing` folder:
`Step1_data_preprocessing_v2.ipynb`: separate the problems based on (1)whether the problems is benchmarked or not and (2)whether there is a user grading.
`Step2_BetaMove.ipynb`: This program first computes the success score of each move by the relative distance between holds and the difficulty scale of each hold. It then finds the best route using beam search algorithm.
`Step3_partition_train_test_set_v2.ipynb`: This program divide the dataset into training/dev/test sets.

The final files that are used for the training and evaluation of GradeNet are `training_seq_n_12_rmrp0`, `dev_seq_n_12_rmrp0`, and `test_seq_n_12_rmrp0`.

The final files that are used for the training of DeepRouteSet are `nonbenchmarkNoGrade_handString_seq_X`, `benchmark_handString_seq_X`, `benchmarkNoGrade_handString_seq_X`, and `nonbenchmark_handString_seq_X`.

### `model`
This folder contains 3 files that are critical to run the experiment and repeat our results.

#### GradeNet

To run GradeNet, open the jupyter notebook in `\model\GradeNet.ipynb` and follow the instructions. You can either re-run the experiments, or load the pretrained weights `\model\GradeNet.h5`.

#### DeepRouteSet

To run GradeNet, open the jupyter notebook in `\model\DeepRouteSet_v4.ipynb` and follow the instructions. You can either re-run the experiments, or load the pretrained weights `\model\DeepRouteSetMedium_v1.h5`. The code in this file is largely modified from a Coursera problem exercise "Improvise a Jazz Solo with an LSTM Network", which is originally modified from https://github.com/jisungk/deepjazz.

#### Predict the grade of generated problems

To evaluate the generated problems, open the jupyter notebook in `\model\Evaluate_Generated_Output_v3.ipynb` and follow the instructions. Please remember to check if `raw_path` is correct.


### `out`
This folder contains our generated problems. The folder `\out\DeepRouteSet_v1` contains style prediction results of our generated data. Those style predictions are very preliminary, and please ignore that part.

### `website`
This is a static website that shows the 65 generated MoonBoard problems using DeepRouteSet. The link to the website: https://jrchang612.github.io/MoonBoardRNN/website/.

The layout of this website is modified from https://github.com/andrew-houghton/moon-board-climbing.

## Potential future items
* StyleNet