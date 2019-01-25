---
layout: post
title:  "Deciphering FrozenLake Environment"
date: '2018-12-25 15:25:00'
---

### Introduction
Welcome to a new post about AI in R. In this post we are going to solve another *simple* AI scenario included in the OpenAI Gym, the FrozenLake. FrozenLake in a maze-like environment and the goal is to escape from it. The environment is a representation of a frozen lake full of holes, the agent have to go from the starting point <code>(S)</code> to the ending point <code>(G)</code> avoiding the holes <code>(H)</code>. The trick is that the frozen tiles <code>(F)</code> don't let the agent move accurately, so each time an agent performs one action there is a chance that it ends up moving into an unwanted direction.  
The environment is finished when the agent reaches the <code>(G)</code> tile (Reward=1) or when it falls into a hole (Reward=0).{: style="text-align: justify"}

<!--more-->
The FrozenLake is represented as a 4x4 grid with the positions numbered from 0 to 15. The <code>(S)</code> is located in the position number 0, the <code>(G)</code> in the position number 15 and the holes are located in positions 5, 7, 11 and 12:
{: style="text-align: justify"}
<pre><code>SFFF
FHFH
FFFH
HFFG</code></pre>

### Results
The efficiency of the random policy solving the FrozenLake is around 0.5%.

![RPagent](/images/FrozenLakeTraining.gif)   
Random policy example

It might look too low, but indeed it's much higher than the efficacy of the random policy in the CartPole environment. However, there is a good thing about the random policy, we can use it to extract the information about actions, failures and rewards to train coming agents so they can learn from it. To do that we need to create a decision tree in which we define the falling probabilities for all combinations of positions-actions.  

![Tree](/images/P3Tree.jpg)

We create the such tree with the following code 

<pre><code>#Growing a tree
SpaceTree<-expand.grid(c(0:16),c(0:3),c(0,1))
colnames(SpaceTree)<-c("Position","Action","Fail")
SpaceTree$N<-0
SpaceTree$Prob<-0
SpaceTree$Reward<-0
SpaceTree$Length<-0
 </pre></code>

The tree is going to account for the falling probability (*SpaceTree$Prob*), the probability of getting a reward when performing that action (*SpaceTree$Reward*), the number of times the agent has visited that tile (*SpaceTree$N*) and how many steps the agent takes until the episode ends (*SpaceTree$Length*).

Next we fill those parameters with the values obtained by the *random-policy* agent stored in the dfGod variable (check the complete script here)

<pre><code>
for (i in 1:nrow(SpaceTree)) {
  SpaceTree$Prob[i] <- length(dfGod2$Position[dfGod2$Position==SpaceTree$Position[i] &
                                                dfGod2$Action==SpaceTree$Action[i] & 
                                                dfGod2$Fail==SpaceTree$Fail[i]])/
                        length(dfGod2$Position[dfGod2$Position==SpaceTree$Position[i] &
                                                dfGod2$Action==SpaceTree$Action[i]])
  
  SpaceTree$N[i] <-length(dfGod2$Position[dfGod2$Position==SpaceTree$Position[i] &
                                               dfGod2$Action==SpaceTree$Action[i] & 
                                               dfGod2$Fail==SpaceTree$Fail[i]])

  SpaceTree$Reward[i] <- length(dfGod2$Position[dfGod2$Position==SpaceTree$Position[i] &
                                               dfGod2$Action==SpaceTree$Action[i] & 
                                                 dfGod2$RewardEp==1])/
                      length(dfGod2$Position[dfGod2$Position==SpaceTree$Position[i] &
                                              dfGod2$Action==SpaceTree$Action[i]])
  
  SpaceTree$Length[i]<-mean(dfGod2$StepMax[dfGod2$Position==SpaceTree$Position[i] &
                                             dfGod2$Action==SpaceTree$Action[i]&
                                             dfGod2$RewardEp==1])
  
}
</code></pre>

Now that we have the *library* of actions-fails-rewards we can create several policies so the agent look for the better option based on previous oservations:

***Reward based policy:*** 
<pre><code>
      #Action driven by Reward
      PosAct<-subset(SpaceTree, SpaceTree$Position==dfGodN$Position[j] & SpaceTree$Fail==0)
      action <- subset(PosAct,PosAct$Reward == max(PosAct$Reward))
      action <- action$Action
      if (length(action) > 1) action<- action[round(runif(1,min = 1, max = length(action)))]</code></pre>

This agent searches the probability of solving the environment for each combination of position/action and performs the action in which this probability is the highest. In the case that two or more positions/actions have the same probability of solving the environment the agent randomly selects one of them.
This agent solves the enviroment in **15.4%** of the attempts (30 times better than the random policy).

***Hole avoidance policy:***
<pre></code>
      #Action driven by hole avoidance
      PosAct<-subset(SpaceTree, SpaceTree$Position==dfGodN$Position[j] & SpaceTree$Fail==1)
      action <- subset(PosAct,PosAct$Prob == min(PosAct$Prob))
      action <- action$Action
      if (length(action) > 1) action<- action[round(runif(1,min = 1, max = length(action)))]</code></pre>

This agent performs the action in which the probability of falling into a hole is lowest, independetly if the agent solves the enviroment at the or not This agent also includes the random selection of actions if two or more actions have the same probability of not falling. 
The efficiency of this avoidance policy is around **42%**. It makes sense that this agent performs better than the reward-based agent because the random-policy obtained much more information about actions that lead to a fall in the hole than combinations of actions that solve the environment.        

***Reward-Hole policy:***
<pre><code>
       PosAct<-subset(SpaceTree, SpaceTree$Position==dfGodN$Position[j] & SpaceTree$Fail==1)
       action <- subset(PosAct,PosAct$Prob == min(PosAct$Prob))
       action <- action$Action[action$Reward == max(action$Reward)]
       if (length(action) > 1) action<- action[round(runif(1,min = 1, max = length(action)))]</code></pre>
       
Based on the results obtained by the *hole-avoider* and the *reward-searcher* agents I decided to combine their policies to see if it is possible to increase the efficiency. This *hole-avoider-reward-searcher* agent, avoid the actions that leaded the random agent to fall in a hole and in case of several actions having the same minimal probability of falling the agent selects the one with the highest probability of solving the episode. This agent performs surprinsingly well, it solves the enviroment **73.4%** of the times. However I noticed that the agent took a lot of steps to solve the environment (41 steps on average), by introducing a delay in the execution of the actions with <code>Sys.sleep()</code> I could explore the behaviour of the agent:

![Stucked](/images/Stucked.gif)

The agent gets stuck in the beggining of the episode and it seems that it can only continue when it randomly ends in any unwanted tile. I think of this behaviour as if the were agent pushing the borders of the enviroment to avoid falling, which isn't a bad solution to avoid falling.
But what if we prioritize the solution of the environment over the probability of failing? Until now the agent makes the destitions based first on the probability of not falling in a hole and in case of doubt it checks the final reward. Lets invert the two steps of the algorithm: 

<pre><code>
      PosAct<-subset(SpaceTree, SpaceTree$Position==dfGodN$Position[j] & SpaceTree$Fail==1)
      action <- subset(PosAct,PosAct$Reward == max(PosAct$Reward))
      action <- PosAct$Action[PosAct$Prob == min(PosAct$Prob)]
      if (length(action) > 1) action<- action[round(runif(1,min = 1, max = length(action)))]</code></pre>

This agent shows a decent performance although is not as good its sibling's, only **41.2%** of the times partialy inheriting the erratic behaviour observed in its sibling.

***Reward-Speed policy***
<pre><code>
       action <- subset(PosAct,PosAct$Reward == max(PosAct$Reward))
       action <- action$Action[action$Length == min(action$Length)]
       if (length(action) > 1) action<- action[round(runif(1,min = 1, max = length(action)))]</code></pre>

As a solution to fix the erratic behaviour observed in some of the agents I implemented an agent performing a Reward-Speed policy; the agent tries to maximize the reward and then tries to minimizes the number of steps taken. Unfortunately this agent only solves the episode 16.7% of the times. This is probably due to the lack of successful experiences in the library generated by the *random agent*.

Until this point we have seen how it is possible to optimize the use of the library that the random agent obtains; however, none of this agents can really *learn* anything, they need a library of actions and this library could be incorrect, could be outdated or could be very poor in rewared episodes. Additionally it could also be possible to have a library larger that what it's necessary. We need an agent that learns as it experiences the enviroment.

***The self-learner agent***
A self-learner agent needs to store the information that it gets during the different episodes. To do that I have implemented a tree/library at end of every episode to store all the adquired information during previos episodes. The structure of the tree is mainly the same as the one implemented in the other agents, but in order for the agent to learn as it *understands* the environment we need a *learning curve* to make decisions. If the agent hasn't visited enough the current position it executes a random agent so the more experiences about a position the agent has the more it relies on its library of experiences.
I have defined the following learning curve:
![LearningCurve](/images/Learningcurve.jpg)

There are three parameters that can be adjusted:
Slope: How fast the agent transitions from not trusting their experiences to trust them. 
Curiosity50: The number of observations for a position that leads to a 50% chances of trusting the tree/library.
Min: Basal level. The percentage of performing a random action no matter how many observations the agent had.

This type of learning curve allows the agent to adapt as it experiences the enviroment trusting more and more its past observations. Interestingly the inclusion of a basal level would allow for the agent to adapt to a changing enviroment (some adjustments could be included to reset the *library of experiences* if necessary).

I implemented the learning curve as a function, and set the default parameters:

<pre><code>#Learning Curve

learn.curve <- function(x, min, slope, L50){
  min+((1-min)/(1+exp(-slope*(L50-x))))
}

min<-0.05
slope<-0.05
L50<-100

randomness <- learn.curve(1:250,min,slope,L50)
plot(randomness, type = "l", ylim = c(0,1), xlab = "Observations")</code></pre>

![LearningCurveR](/images/learningcurveR.png)

Then to make it have any effect we need to include a control parameter at the end of the action loop to overwrite the selected action with a random one according the the learning curve:

<pre><code>Curiosity <- learn.curve(experienceN,min,slope,L50)
    if(Curiosity > runif(1,min =0 , max= 1) ){
    action <- env_action_space_sample(server, instance_id)
    }</code></pre>

And that's all. We have implemented a simple algorithmic solution for the agent to *learn*. 

To test the *self-learner agent* we calculate the percentage of solved episodes as the number of episodes increase:

![Learner](/images/learner.png)

As you can see the agents starts to increase its performance just after running few iterations. 

### Conclusions






       


