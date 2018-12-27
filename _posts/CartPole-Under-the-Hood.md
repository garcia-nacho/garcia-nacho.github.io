---
layout: post
title:  "CartPole, Under the Hood"
date: '2018-12-25 15:25:00'
---


### Introduction.
In [my last post](https://garcia-nacho.github.io/AI-in-R/) I was explaining how to use AI to solve the [CartPole environment](https://gym.openai.com/envs/CartPole-v0/) by testing sets of 10 radom values as weights of a very simple neural network. In this post we are going to investigate a bit more on what is going on during these processes.

### Distribution of valid sets of weights.
In my previous post about the CartPole I showed that random search of weights is strong enough to solve the enviroment in few rounds. That means that the range of weights that 

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