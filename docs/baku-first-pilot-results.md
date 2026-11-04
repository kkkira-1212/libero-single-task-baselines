# First BAKU pilot results

### First BAKU pilot completed — 0/20 successes
- Training14386819 COMPLETED0:0 in11m50s,onQuadroRTX6000,all3000updates. Fixed32samplevalidationactorloss21.5776initial->0.368096final(nonmonotonic,minlogged0.112909at1500);finalcheckpoint selected as prespecified.
- Eval14386820 COMPLETED0:0 in16m47s;all20trialsIDs/seeds20..39completed,allfailedat600steps. Success0/20=0%,Wilson95%0–16.11%.20GIFs+20actionfilesverified.
- This is a functioning execution pipeline with unsuccessful learned behavior; BAKU effective-baseline milestone NOT achieved. Do not equate loss decrease/jobcompletion with reproduction.
- Negativeactorlossvaluesarepossibleforcontinuous-densityNLL atfixedstd0.1;not inherently a trainingfailure. Thisloss isnot comparable toACTL1/DPnoiseMSE.
- Root cause not established. Before blindly addingseeds/updates,inspectrepresentativevideos,training-vs-onlineact/parity,checkpointweights andtemporalaggregation;then controlleddiagnostic/longertraining as warranted.3000updates isfarshorterthanupstreamsuite's601000steps,butthat alone isnot proof ofundertraining.
- Preservefailedpilotcheckpoint/config/results;nonewjobssubmitted during this review.
