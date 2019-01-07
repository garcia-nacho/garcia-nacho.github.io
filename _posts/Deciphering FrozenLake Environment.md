---
layout: post
title:  "Deciphering FrozenLake Environment"
date: '2018-12-25 15:25:00'
---

### Introduction
Welcome to the second post about AI in R. In this post we are going to solve another simple AI included in the OpenAI Gym, the FrozenLake. FrozenLake in a maze-like environment and the goal is to escape from it. The environment is a representation of a frozen lake full of holes so the agent must go from the starting point to the ending point evading the holes. FrozenLake is represented by a grid of 4x4 positions numbered from 0 to 15. The starting point is located in the position number 0 and the goal in the position number 15 and the holes are located in positions 5, 7, 11 and 12:
<pre><code>
SFFF
FHFH
FFFH
HFFG</code></pre>
S: Starting tile
F: Frozen tile
H: Hole
G: Goal




The efficiency of the random policy is around 0.5%. It might look too low, but it's much higher than the efficacy of the random policy in the CartPole environment. But the good thing of the random policy is that we can extract of the information regarding the failures and reward so the coming agents can learn from them. 

Hole avoidance policy:
<pre></code>
      #Action driven by hole avoidance
      PosAct<-subset(SpaceTree, SpaceTree$Position==dfGodN$Position[j] & SpaceTree$Fail==1)
      action <- subset(PosAct,PosAct$Prob == min(PosAct$Prob))
      action <- action$Action
      if (length(action) > 1) action<- action[round(runif(1,min = 1, max = length(action)))]</code></pre>

The efficiency of the hole avoidance policy is around 42%      

Reward based policy: 
<pre><code>
      #Action driven by Reward
      PosAct<-subset(SpaceTree, SpaceTree$Position==dfGodN$Position[j] & SpaceTree$Fail==0)
      action <- subset(PosAct,PosAct$Reward == max(PosAct$Reward))
      action <- action$Action
      if (length(action) > 1) action<- action[round(runif(1,min = 1, max = length(action)))]</code></pre>

The efficiency of the Reward-based policy is 15.4% (30 times better than the random policy), probably because the number of solved episodes by the random policy is too low. 9 steps on average for the reward based policy

Reward-Hole policy:

<pre><code>
       PosAct<-subset(SpaceTree, SpaceTree$Position==dfGodN$Position[j] & SpaceTree$Fail==1)
       action <- subset(PosAct,PosAct$Prob == min(PosAct$Prob))
       action <- action$Action[action$Reward == max(action$Reward)]
       if (length(action) > 1) action<- action[round(runif(1,min = 1, max = length(action)))]</code></pre>
       






The average length of an episode is 7.6 steps while the average lengh of an episode under the hole avoidance policy is 35 steps


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




