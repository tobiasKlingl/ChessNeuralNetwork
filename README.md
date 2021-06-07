# Fun project: ChessNeuralNetwork

## PlayingCode_1_3: Used to simulate games
- How to use the code:
*    cd PlayingCode_1_3
*    source run.sh ( or: source run.sh >& log & )

- This will generate the training data for the network -> Stored in
*    PlayingCode_1_3/data/training_data_iteration_${SOMENUMBER}.pkl
- The number of games to play and the maximum number of plys can be set in run.sh
- The gameMode (random vs. random, network vs. random, network vs. network) can be set in run.sh
* For 'network vs. random' or 'network vs. network' a neural network need to be stored in
*    Chess/NetworkCode/saves/iteration_${SOMENUMBER}/${FileName}
*  When playing with the network (while useMultiprocessing=True in playGame.py) change "import cupy as cp" -> "import numpy as cp" in NetworkCode/NNlib_1_7.py
- For debugging purposes it is possible to play yourself. Possible modes are: "WhiteVsBlack", "WhiteVsComputer","BlackVsComputer"
- The amount output information displayed can be controlled in playGame.py:
*    debug: if true -> detailed information 
*    noOutputMode: if true -> only very few output is shown
*    printMergeInfo: if true -> show information about the merging process to obtain the percentages to win from a given position

## NetworkCode: Used to train the neural network
- How to use the code:
*    cd NetworkCode
*    python run_nn_1_7.py
   
- If cupy is not available on your system change "import cupy as cp" -> "import numpy as cp" in
  NetworkCode/NNlib_1_7.py  and NetworkCode/data_loader.py
- Uses the training data stored in PlayingCode_1_3/data/training_data_iteration_${SOMENUMBER}.pkl to train the network.
* The trained network will be stored in
*    Chess/NetworkCode/saves/iteration_${SOMENUMBER}/${FileName}