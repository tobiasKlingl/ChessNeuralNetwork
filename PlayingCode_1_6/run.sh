#!/bin/bash

nGames=500         #number of games to play from each position
searchDepth=10     #search depth for each position
upToPly=200        #run code until this ply
checkForDoubles=5  #check if position has been played up this ply
numJobs=4          #number of jobs
iteration=0        #neural network iteration id. it=0 => random vs. random
mergeInfo=true

for (( job=0; job<numJobs; job++ )); do
    echo "job, nGames, searchDepth, upToPly, checkForDoubles, iteration: ${job}, ${nGames}, ${searchDepth}, ${upToPly}, ${checkForDoubles}, ${iteration}"
    python playGames.py ${nGames} ${searchDepth} ${upToPly} ${checkForDoubles} ${job} ${iteration} ${mergeInfo} &> logs/log_${job} &
    #python playGames.py ${nGames} ${searchDepth} ${upToPly} ${checkForDoubles} ${job} ${iteration} ${mergeInfo}
    process_id=$!
    echo "Job ${job} started: ID=${process_id}"
done

wait
echo "Done waiting! Now merge the files."
python mergeTrainFiles.py ${iteration} ${mergeInfo}
