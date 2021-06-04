#!/bin/bash

numberOfGamesToPlay=1
StopGameNumber=100
checkForDouble=5
gameMode="SelfplayRandomVsRandom"
#gameMode="SelfplayNetworkVsRandom"
iteration=1

for (( c=0; c<1; c++)); do 
    python3 playGame.py ${numberOfGamesToPlay} ${StopGameNumber} ${checkForDouble} ${gameMode} ${iteration}
    echo "Rename data/training_data_iteration_${iteration}_new.pkl -> data/training_data_iteration_${iteration}.pkl"
    mv data/training_data_iteration_${iteration}_new.pkl data/training_data_iteration_${iteration}.pkl 
done
