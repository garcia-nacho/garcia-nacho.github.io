---
layout: post
title:  "CartPole, Under the Hood"
date: '2018-12-25 15:25:00'
---


### Introduction.
In [my last post](https://garcia-nacho.github.io/AI-in-R/) I was showing how to use AI to solve the [CartPole environment](https://gym.openai.com/envs/CartPole-v0/) by assigning and testing sets of 10 random values as weights of a very simple neural network. In this post, we are going to investigate a bit more on what is going on during these processes. 
In machine learning, the weights are the internal parameters that are used to transform the input values into the predictions. The weights are usually initiated in a random manner and tuned during the training process so the predictions are more and more accurate. We can say that the training process is indeed the adjustment of the weights of the model. In our neural network, we don't really tune them, we randomly assign values to the weights and check the performance of the agent. This strategy only works if the space of working weights is big enough, so it is very likely that we can find them just by chance.   

### Distribution of valid sets of weights.
In [our model](https://github.com/garcia-nacho/ArtificialIntelligence/blob/master/ExplorationRandomSearchNN.R) the chance of getting a working model is that high that the first time that I run the model I obtained a valid model able to solve the enviroment after 3 rounds. However, how likely is to find a valid set of weights? To answer this question I modified the script to it doesn't break when the solved condition is acchieve  

### Neural network vs linear model.
In my last post I mentioned that the CartPole environment could be solved finding the weights (and the intercept) of a linear model such like this:
<pre><code>
Action = Obs1 * Weight1 + Obs1 * Weight1 +Obs1 * Weight1 +Obs1 * Weight1 + Intercept 
</code></pre>
But this type of linear models (without the intercept) are indeed an special case of a neural network:

### Generation of novel neural networks able to solve the CartPole

Until this point I have shown that 


### Neural networks seem to be more stable in noise conditions.

<pre><code>
Noise<-0.7
    for (l in 1:Observations) {
    dfEyes[j,l]<-dfEyes[j,l]+(runif(1, min = -Noise, max = Noise)*dfEyes[j,l]) 
    }
    Input<-dfEyes[j,1:Observations]
</code></pre>

LinReg No Noise: (66%) N=3
LinReg Noise 0.7: (0%)

NNet No Noise: (69%) N=36
NNet noise 0.7 (25%)

Everything seems to suggest that neural networks provide more stability to the agent than linear regressions. However the number of observations is too low to conclude it without any doubt so more simulations would be needed.
