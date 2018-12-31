---
layout: post
title:  "CartPole, Under the Hood"
date: '2019-01-01 15:25:00'
---


### Introduction.
In [my last post](https://garcia-nacho.github.io/AI-in-R/) I was showing how to use AI to solve the [CartPole environment](https://gym.openai.com/envs/CartPole-v0/) by assigning and testing sets of 10 random values as weights of a very simple neural network. In this post, we are going to investigate a bit more on what is going on during these processes. 
In machine learning, the weights are the internal parameters that are used to transform the input values into the predictions. The weights are usually initiated in a random manner and tuned during the training process so the predictions are more and more accurate. We can say that the training process is indeed the adjustment of the weights of the model. In our neural network, we don't really tune them, we randomly assign values to the weights and check the performance of the agent. This strategy only works if the space of working weights is big enough, so it is very likely that we can find them just by chance.   

### Distribution of valid sets of weights.
In [our model](https://github.com/garcia-nacho/ArtificialIntelligence/blob/master/ExplorationRandomSearchNN.R) the chance of getting a working model is that high that the first time that I run the model, I obtained a valid set of weights able to solve the enviroment just after 3 rounds. However, how likely is to find a valid set of weights? To answer this question I modified the script so it doesn't stop when the environment is solved so we can count the number of valid models at the after N rounds.
After 1500 rounds I obtained this distribution of successes:
![P3Density](/images/P3Density.png)
In which almost a 3% of the combinations of weights are able to solve the environment. However as you can see from the plot we have created a large range of agents, most of them don't work, some of them perform over a random-agent and some of them solve the environment.  

### Neural network vs linear model.
In my last post I already mentioned that the CartPole environment could be solved finding the weights (and the intercept) of a linear model such like this:<pre><code>
Action = Obs1 * Weight1 + Obs2 * Weight2 + Obs3 * Weight3 + Obs4 * Weight4 + Intercept 
</code></pre>
However this type of linear models (without the intercept) belong indeed to special case of a neural network in which the weight connecting one neuron with the output (*WN2.2*) is 0 and the other weight of the other neuron is 1 (*WN2.1*):
![P3Scheme](/images/P3Scheme1.jpg)

So I wanted to explore the weights, specially the distribution of *WN2.2* and *WN2.1* to see if it was somehow related with the ability of the agent to solve the CartPole.   
![P3LinReg](/images/P3LinReg.png)
V9 is the weight *WN2.1* and V10 is *WN2.2*, in blue are represented those values of *WN2.1* and *WN2.2* that are able to solve the CartPole (together with the other 8 weights). The triangles represent those values of WN2.1 and WN2.2 in which our neural network approximates a linear regression model (very high or low ratios *WN2.1* / *WN2.2*).   
First of all, it seems that all range of *WN2.1* and *WN2.2* are permited in the agents able to solve the CartPole. It also seems that some of the solving agents (3 out of 39) have a linear regression policy instead of a neural network. So the next question is if the NNet agents perform differently from the LinReg ones in any condition.

### Neural networks seem to be more stable in noise conditions.
Just from a visual inspection of the episodes executed by different agents we can find that even though all of them solve the environment there are differences in the behaviour of them.
![LinRegCartPole](/images/LinRegCartPole.gif)
*LinReg-Agent-1*

![LinRegNNet](/images/NNetCartPole.gif)
*NNet-Agent-1*

If you carefully look at how the pool balances, it becomes clear that the two agents behave differently. That means that there might be differences in the performace of different agents, unfortunately there is no easy way to run the environment for more than 200 steps without touching the Python code of the gym, so it is not easy to find out how the different agents would perform in longer episodes. However, we could introduce a noise parameter to evaluate the performace of the agents in a noisy environment comparing them. Noise, in this context, can be undestood as a random modification of the observations that the agent observes. I have implemented the noise as follows:
<pre><code>
#Noise level up to 70%
Noise<-0.7
    for (l in 1:Observations) {
    dfEyes[j,l]<-dfEyes[j,l]+(runif(1, min = -Noise, max = Noise)*dfEyes[j,l]) 
    }
    Input<-dfEyes[j,1:Observations]
</code></pre>
The agent perceives the enviroment with a distortion in all observation up to a 70% (added or substracted). We can think of this type of noise a real-world situation in which the sensors connected to the agent are faulty or imprecise. This type of situations is really common in some delicate computer-assisted operations such as those carried out in the aerospacial field (you can read more about this [here](https://ieeexplore.ieee.org/document/5466132).

I re-evaluated the performance of the *LinReg* (n=3) and *NNet* (n=36) agents under this noisy conditions and I found that none of the the *LinReg* agents were able to solve the environment, however almost half of the *NNet* agents were still able to solve the noisy environment:

![LinRegNoise](/images/LinRegNoise.gif)
*LinReg agent #1* + Noise

![NNetNoise5](/images/NNetNoise5.gif)
*NNet agent #5* + Noise

Everything seems to suggest that neural networks based policies provide more stability to the agent than linear regressions ones. However the number of observations is too low to conclude it without any doubt so more simulations and simulations in which *WeightN2.1* and *WeightN2.2* are forced to 1 and 0 respectively. 

Anyhow it seems to be clear that there is a lot of approaches for the different agents to solve the environment and some of them more resistant to noise interferences. 

### *Artificial* Generation of Agents Able to Solve the CartPole

Since the weights are what define an agent, we could say that they would represent *"the software of the sotfware"* and the neural network would be the *"hardware of the software"*. Until this point we have seen that the different agents loaded with different versions of the *"software"* peform distinctly. Therefore it would be very interesting to expand the number of agents so we could study and maybe rank them. In this section I am going to show you an iterative approach able to generate thousands of agents in just few steps.

It is logical to expect that all working sets of weights have internally something in common, meaning that there is an internall hidden relationship between the 10 weights that conform a valid set, and this is the hypothesis that we are going to test. To do that we are going to implement another deep neural network in which we are going to use the 10 weights as inputs and the 

We are going to train the network using the 1500 sets of weights that we already obtained. In this case we are going to use a standalone library that we can access from R, h2o. I like h2o because it is simple but pretty flexible, you can implement deep neural networks, random forests, gradient boosting machines, linear regressions or model ensembles among others. It also has a function to perform hyperparameter searches, which make it a very complete library for basic (and not that basic) machine learning tasks.

The goal of this post is not talk about h2o how to install it or how to run it. You can read about those topics [here](http://docs.h2o.ai/h2o/latest-stable/h2o-docs/welcome.html). 

I implemented a clasifier using a basic deep neural net whith two internal layers of 24 neurons (I was playing a little bit with the architectures and 24-24 performed fairly well. 
<pre><code 
#h2o initialization
h2o.init()

#Transforming the output variable into a binary columns with two factors
dfWeights$Completed[dfWeights$Completed=="YES"]<-1
dfWeights$Completed[dfWeights$Completed=="NO"]<-0
dfWeights$Completed<-as.factor(dfWeights$Completed)

#Split the dataset into training and testing to validate the performance
dfWeightsTrain<-dfWeights[1:900,]
dfWeightsTest<-dfWeights[901:1000,]

#Load the datasets into h2o
df.h2oTrain <- as.h2o(dfWeightsTrain)
df.h2oTest <- as.h2o(dfWeightsTest)

  NNet    <-         h2o.deeplearning(x= 1:10,
                     y=12,
                     stopping_metric="logloss",
                     distribution = "bernoulli",
                     loss = "CrossEntropy",
                     balance_classes = TRUE,
                     validation_frame = df.h2oTest,
                     training_frame	=	df.h2oTrain,
                     hidden = c(24,24),
                     nfolds = nfolds,
                     fold_assignment = "Modulo",
                     keep_cross_validation_predictions = TRUE,
                     epochs = 5) </code></pre>
                 
After training I checked the performance on the test set



further studies could be oriented to infer the non linear relationships between weights so we could understand it. 


As you can see there is endles fun under the hood of AI implementations and the more control you have over your algorithms and functions the more you can explore. 



![P3Clones.gif](/images/P3Clones.gif)
***"The best thing about being me, there's so many me's"***
*Agent Smith (The Matrix Reloaded, 2003)*

### Notes
After testing to fit a linear regression model to a 
When I say h2o is used for basic machine learning task I am talking about task that do not require advance modifications of the model. However h2o is used in many production environments in which the basic analyses are enough. 
