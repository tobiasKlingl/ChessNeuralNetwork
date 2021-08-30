#!/bin/bash

numberOfGamesToPlay=10
StopGameNumber=200
checkForDoubles=5
gameMode="SelfplayRandomVsRandom"
#gameMode="SelfplayNetworkVsRandom"
iteration=1

for (( plyToStart=0; plyToStart<1; plyToStart++)); do 
    python3 playGame.py ${numberOfGamesToPlay} ${StopGameNumber} ${gameMode} ${iteration} ${plyToStart} ${checkForDoubles}
    exitState=$?
    echo "Rename data/training_data_iteration_${iteration}_new.pkl -> data/training_data_iteration_${iteration}.pkl"
    mv data/training_data_iteration_${iteration}_new.pkl data/training_data_iteration_${iteration}.pkl
    echo "exitState=$exitState"
    if [[ $exitState -eq 1 ]]; then
    	echo "exitState = $exitState ! Breaking loop!"
    	break
    fi
done
