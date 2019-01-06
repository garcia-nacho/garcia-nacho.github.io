---
layout: post
title:  "Deciphering FrozenLake Environment"
date: '2018-12-25 15:25:00'
---


Lets calculate the efficiency of the random policy almost 0.55%. It might look too low, but it's much higher than the efficacy of the random policy in the CartPole environment.


The tree policy 

The average length of an episode is 7.758 steps while the average lengh of an episode under the tree policy is 35.556 steps. That means that the efficiency is too low. 

The efficiency of the tree policy 0.63%, slightly higher but still very far from solving the environment. The main problem is indeed that the agent spends too much time in a "dangerous" environment and this happens because there are several situations in which the agent is forced to make a random decision since the probabily of falling in a hole is 0 instead doing the optimal movement. To solve it I have included another extra step, once the agent encounters several actions with the same probabilty of fail in checks which one has a higher probability of solving the environment and executes that one. 

<pre><code>
      action <- subset(PosAct,PosAct$Prob == min(PosAct$Prob))
      action <- PosAct$Action[PosAct$Reward == max(PosAct$Reward)]
      if (length(action) > 1) action<- action[round(runif(1,min = 1, max = length(action)))] </code></pre>


Now the agent is able to solve the environment in 12% of the episodes and the episode lenght has been reduced to 10.22 steps.

Few questions what if we prioritize the solution of the environment over the probability of failing? Until now the agent makes the destitions based first on the probability of not falling in a hole and in case of doubt it checks the final reward. Lets invert the two steps of the algorithm: 

<pre><code>
      action <- subset(PosAct,PosAct$Reward == max(PosAct$Reward))
      action <- PosAct$Action[PosAct$Prob == min(PosAct$Prob)]
      if (length(action) > 1) action<- action[round(runif(1,min = 1, max = length(action)))]</code></pre>

Now the agent solves the environment 41.2% of the times with tan average episode lenght of 37.1 steps




