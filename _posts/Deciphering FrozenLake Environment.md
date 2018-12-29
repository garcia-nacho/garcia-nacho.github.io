---
layout: post
title:  "Deciphering FrozenLake Environment"
date: '2018-12-25 15:25:00'
---


Lets calculate the efficiency of the random policy   
<rep><code>length(dfGod$V1[which(dfGod$Done==TRUE & dfGod$Reward==1)])/length(unique(dfGod$Round))
[1] 0.01866667 </code></rep>
almost 2%. It might look too low, but it's much higher than the efficacy of the random policy in the CartPole environment.

