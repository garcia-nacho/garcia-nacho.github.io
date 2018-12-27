---
layout: post
title:  "CartPole, Under the Hood"
date: '2018-12-25 15:25:00'
---


### Introduction.
In [my last post](https://garcia-nacho.github.io/AI-in-R/) I was explaining how to use AI to solve the [CartPole environment](https://gym.openai.com/envs/CartPole-v0/) by testing sets of 10 radom values as weights of a very simple neural network. In this post we are going to investigate a bit more on what is going on during these processes.

### Neural network vs linear model.
In my last post I mentioned that the CartPole environment could be solved finding the weights (and the intercept) of a linear model such like this:
<pre><code>
Action = Obs1 * Weight1 + Obs1 * Weight1 +Obs1 * Weight1 +Obs1 * Weight1 + Intercept 
</code></pre>
But this type of linear models (without the intercept) are indeed an special case of a neural network:
