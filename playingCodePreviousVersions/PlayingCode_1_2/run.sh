#!/bin/bash

numberOfGamesToPlay=1
StopGameNumber=200
#gameMode="SelfplayRandomVsRandom"
gameMode="SelfplayNetworkVsRandom"
iteration=1

for (( plyToStart=0; plyToStart<1; plyToStart++)); do 
    python3 playGame.py ${numberOfGamesToPlay} ${StopGameNumber} ${gameMode} ${iteration} ${plyToStart}
    echo "Rename data/training_data_iteration_${iteration}_new.pkl -> data/training_data_iteration_${iteration}.pkl"
    mv data/training_data_iteration_${iteration}_new.pkl data/training_data_iteration_${iteration}.pkl 
done
