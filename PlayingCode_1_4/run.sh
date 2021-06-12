#!/bin/bash

nGames=500         #number of games to play from each position
StopGameNumber=999 #run each game until this ply
upToPly=500        #run code until this ply
checkForDoubles=5  #check if position has been played up this ply
numJobs=4          #number of jobs
#simul=4           #number of parallel jobs
iteration=0        #neural network iteration id. it=0 => random vs. random
mergeInfo=true

for (( job=0; job<numJobs; job++ )); do
    echo "job, nGames, StopGameNumber, upToPly, checkForDoubles, iteration: ${job}, ${nGames}, ${StopGameNumber}, ${upToPly}, ${checkForDoubles}, ${iteration}"
    python playGames.py ${nGames} ${StopGameNumber} ${upToPly} ${checkForDoubles} ${job} ${iteration} ${mergeInfo} &> logs/log_${job} &
    process_id=$!
    echo "Job ${job} started: ID=${process_id}"
done

wait
echo "Done waiting! Now merge the files."
python mergeTrainFiles.py ${numJobs} ${iteration} ${mergeInfo}
