---
layout: post
title:  "Finding anomalies in other stars"
date: '2019-02-18 19:25:00'
---

### Introduction.
When it comes to answering one of the most relevant questions that humanity has still open, we need to admit that we don't have a clue and this question is: **"Are we alone in the universe?"**.   
The first historical references to the question are found in ancient Greece. Greek and Roman philosophers reasoned that since the universe seemed to be so vast it would be reasonable to think that other "tribes" would be living somewhere out there. Later on, religious thinking displaced the logical thinking, and humanity was special just because it was God's will to be special. And it was not until the Renascent when others started to ask the same question. If the Earth is no the center of the universe and there are other planets and stars, why should be life on Earth special?   

If you think for a second on the implications that answering that question have, you will realize how crucial it is for us, humans, to answer that question. It doesn't really matter the direction of the answer brecause just knowing with 100% certainty that there is or there is not life somewhere else in the universe would change us as a whole. 

Unfortunately, neither ancient greeks or guys like Copernicus or Huygens had the technology to find it out and indeed we still don't know if we can answer that question today, but today in this post, I will show you some approaches to try to answer that question. 

{: style="text-align: justify"}
<!--more-->

### The paradox of Drake equation
In 1961 Frank Drake tried to shed some light on the topic by turning the philosophical question into a parametric hypothesis. He numerically computed the probability of another intelligent civilization living in our galaxy.  For the first time in history, someone had answer the question from a numerical point of view.  According to Drake, we are sharing our galaxy with another 10 intelligent civilizations with the ability to communicate. Is that too much? is that too little? Well... for sure we would not be able to detect our own presence from a distance 10K light-years (3,1kpc) which is approximately 1/10 of the galactic size. Indeed we would not be able to detect even the presence of Earth since the furthest exoplantet detected and confirmed it is around 6.5K light-years:

[Exoplanets](/images/1920px-Distribution_of_exoplanets_by_distance.png)

But coming back to Drake equations, it takes into account the following parameters:

![DrakeEq](/images/drakeequation.svg)

R∗ = the rate of star formation
fp = the ratio of stars having planets
ne = the ratio of planets in the habitable zone
fl = the ratio of ne that would actually develop life
fi = the ratio of fl that would develop intelligent life
fc = the ratio of civilizations that develop a technology detectable from other planets 
L = the time for which civilizations send detectable signatures to space

Of course, Drake equation is based on very strong assumptions 

{% highlight r %}
linear<-lm(log(df$Infected-1)~df$Day)
start <- list(a=exp(coef(linear)[1]), b=coef(linear)[2])
{% endhighlight %}
