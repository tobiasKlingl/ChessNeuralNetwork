# Fun project: ChessNeuralNetwork  
  
## PlayingCode_${VERSION}: Used to simulate games  
- How to use the code (version 1_3 uses pythons multiprocessing module, version 1_4 is based on numba (tested with numba version 0.51 and 0.53) to improve performance):  
        cd PlayingCode_${VERSION}  
        source run.sh ( or: source run.sh >& log & )  
  
- This will generate the training data for the network -> Stored in:  
        PlayingCode_${VERSION}/data/training_data_iteration_${SOMENUMBER}.pkl  
- The number of games to play and the maximum number of plys can be set in run.sh  
- The gameMode (random vs. random, network vs. random, network vs. network) can be set in run.sh  
    For 'network vs. random' or 'network vs. network' a neural network need to be stored in:  
        Chess/NetworkCode/saves/iteration_${SOMENUMBER}/${FileName}  
- The amount of output information displayed can be controlled in playGame.py:  
        debug: if true -> detailed information  
        noOutputMode: if true -> only very few output is shown  
        printMergeInfo: if true -> show information about the data merging process  
  
## NetworkCode: Used to train the neural network  
- How to use the code (cupy and numba versions are available):  
        cd NetworkCode  
        python run_nn_${VERSION}.py  
   
- In cupy version: If cupy is not available on your system change "import cupy as cp" -> "import numpy as cp" in:  
        NetworkCode/NNlib_1_7_cupy.py  and NetworkCode/data_loader_cupy.py  
- Numba does not support gpu support for jitclass yet => cupy version is faster atm.  
- For the training process the training data stored in PlayingCode_{VERSION}/data/training_data_iteration_${SOMENUMBER}.pkl is used.  
- The trained network will be stored in:  
        Chess/NetworkCode/saves/iteration_${SOMENUMBER}/${FileName}  