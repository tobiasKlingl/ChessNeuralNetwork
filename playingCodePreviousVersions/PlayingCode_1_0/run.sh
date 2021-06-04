#!/bin/bash

numberOfGamesToPlay=1
StopGameNumber=100
checkForDouble=5
gameMode="SelfplayRandomVsRandom"
#gameMode="SelfplayNetworkVsRandom"


for (( c=0; c<1; c++)); do 
    python3 playGame.py ${numberOfGamesToPlay} ${StopGameNumber} ${checkForDouble} ${gameMode}
    echo "rename data/training_data_new.pkl -> data/training_data.pkl"
    mv data/training_data_new.pkl data/training_data.pkl 
done
