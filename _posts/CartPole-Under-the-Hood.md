---
layout: post
title:  "CartPole, Under the Hood"
date: '2018-12-25 15:25:00'
---


### Introduction.
In [my last post](https://garcia-nacho.github.io/AI-in-R/) I was showing how to use AI to solve the [CartPole environment](https://gym.openai.com/envs/CartPole-v0/) by assigning and testing sets of 10 random values as weights of a very simple neural network. In this post, we are going to investigate a bit more on what is going on during these processes. 
In machine learning, the weights are the internal parameters that are used to transform the input values into the predictions. The weights are usually initiated in a random manner and tuned during the training process so the predictions are more and more accurate. We can say that the training process is indeed the adjustment of the weights of the model. In our neural network, we don't really tune them, we randomly assign values to the weights and check the performance of the agent. This strategy only works if the space of working weights is big enough, so it is very likely that we can find them just by chance.   

### Distribution of valid sets of weights.
In [our model](https://github.com/garcia-nacho/ArtificialIntelligence/blob/master/ExplorationRandomSearchNN.R) the chance of getting a working model is that high that the first time that I run the model, I obtained a valid set of weights able to solve the enviroment just after 3 rounds. However, how likely is to find a valid set of weights? To answer this question I modified the script so it doesn't stop when the environment is solved so we can count the number of valid models at the after N rounds.
After 1500 rounds we found this distribution of success:
![P3Density](/images/P3Density.png)
In which atmost a 3% of the combinations of weights are able to solve the environment. However as you can see from the plot we have created a large range of agents, most of them don't work, some of them perform over a random-agent and some of them solve the environment.  

### Neural network vs linear model.
In my last post I already mentioned that the CartPole environment could be solved finding the weights (and the intercept) of a linear model such like this:
<pre><code>
Action = Obs1 * Weight1 + Obs1 * Weight1 +Obs1 * Weight1 +Obs1 * Weight1 + Intercept 
</code></pre>
However this type of linear models (without the intercept) belong indeed to special case of a neural network in which the weight connecting one neuron with the output is 0 and the other weight of the other neuron is 1:


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
