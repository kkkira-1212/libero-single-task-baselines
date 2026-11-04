Hi Shuxin, a quick update: I’ve done preliminary single-task runs of ACT, Diffusion Policy, and BAKU on LIBERO, and have been adapting A2A. I identified an observation–action timing mismatch in the adapter. In a matched experiment, correcting it changed A2A from 0/20 to 20/20 successful rollouts on the same task.

This is still a preliminary result with one training seed and six flow steps. I’ve queued confirmation runs with another training seed, new initial states, and single-step inference.

I have an exam this Thursday (Oct 29), so I’m aiming to consolidate the results by Sunday (Nov 1) and be ready to give a short report on Monday (Nov 2). I’ll include the implementation changes, results, and remaining limitations. Would that timeline work for you?
