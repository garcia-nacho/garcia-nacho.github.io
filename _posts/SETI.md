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

### The Fermi paradox and the Drake equation
In 1961 Frank Drake tried to shed some light on the topic by turning the philosophical question into a parametric hypothesis. He numerically computed the probability of another intelligent civilization living in our galaxy.  For the first time in history, someone had answer the question from a numerical point of view.  According to Drake, we are sharing our galaxy with another 10 intelligent civilizations with the ability to communicate. Is that too much? is that too little? Well... for sure we would not be able to detect our own presence from a distance 10K light-years (3,1kpc) which is approximately 1/10 of the galactic size. Indeed we would not be able to detect even the presence of Earth since the furthest exoplantet detected and confirmed it is around 6.5K light-years:

![Exoplanets](/images/1920px-Distribution_of_exoplanets_by_distance.png)

But coming back to Drake equations, it takes into account the following parameters:

![DrakeEq](/images/drakeequation.svg)

R∗ = the rate of star formation
fp = the ratio of stars having planets
ne = the ratio of planets in the habitable zone
fl = the ratio of ne that would actually develop life
fi = the ratio of fl that would develop intelligent life
fc = the ratio of civilizations that develop a technology detectable from other planets 
L = the time for which civilizations send detectable signatures to space

Of course, some of Drake equation's constrains are purely theoretical, for instance... we don't know how easy it is for a form of life to become intelligent and in that sense, there is a new paper in which the authors develop a theoretical Bayesian framework to estimate some of those parameters and they argue that considering how early life arrived to Earth and how late this life became intelligent[Ref](https://www.pnas.org/content/117/22/11995), it might not be an easy step to take; however, the universe is vast so big that even events with very low probality are likely to happen somewhere and this is where the Fermi paradox comes in. If there is so likely that there is intelligent life somewhere in the observable universe, why haven't we heard from them??!! 
There are so many posible answers to try to explain the paradox that it is impossible to mention them all, and still it is a paradox because none of the arguments is strong enough to solve it.

{: style="text-align: justify"}

### The SETI/Breakthrough programs 
Since we haven't been able to disprove any hypothesis regarding extraterrestrial intelligence, we can only keep scanning the space looking for a signature of an intelligent civilization and this is what the SETI does. Through different initiatives SETI scans most of the electromagenetic spectrum (from radio waves to light sources) trying to find unnatural signals. However the amount of data gathered so far is so vast that a comprenhensive analysis using classical approaches would take a lot of years, so SETI, through the Breakthrough program has released a masive dataset containing data from the closests 100000 stars. The data comes from    [The Parkes Observatory](https://en.wikipedia.org/wiki/Parkes_Observatory), [The Green Bank Telescope](https://greenbankobservatory.org/science/telescopes/gbt/) or the [Automated Planted Finder(Lick Observatory)](https://en.wikipedia.org/wiki/Automated_Planet_Finder) and you can download everything from [here](https://breakthroughinitiatives.org/opendatasearch).
The idea behind releasing such a dataset is to make it available for the community, so everyone can explore to try to find signatures indicative of alien civilizations, and this what I will do in this post:to establish a framework and to provide some ideas so everyone can explore the dataset. 

### Navigating the data. 
In this first post on the topic, I will focus on the APF dataset because of it's simplicity. The APF dataset is a set of images in XXX format. Each image corresponds to the observation of one star at one particular timepoint. The "images" are indeed spectrograms. This spectrograms are generated by a telescope coupled with an [echelle spectrogram](https://en.wikipedia.org/wiki/Echelle_grating) which splits a the light coming from an star into its different wavelenghts. This allows us to measure the number of photons of each wavelength coming from that particular start:
Here is an scheeme of an echelle spectograph:

![Echelle](/images/micromachines-10-00037-g002.png)  [Source](https://www.mdpi.com/2072-666X/10/1/37/htm)

The raw images that we obtain from the echelle telescope look like this:

![Echelle image](/images/echelle1.png)

And this is how it looks when we zoom in:

![Echelle zoom](/images/echelle%20zoom.png) 



https://www.pnas.org/content/115/42/E9755




{% highlight r %}
linear<-lm(log(df$Infected-1)~df$Day)
start <- list(a=exp(coef(linear)[1]), b=coef(linear)[2])
{% endhighlight %}
