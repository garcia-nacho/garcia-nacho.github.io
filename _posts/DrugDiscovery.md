---
layout: post
title:  "Computational Drug Discovery (Novice Mode)"
date: '2019-08-25 21:30:00'
---

## Introduction.
We humans learned how to use drugs many thousands of years ago. The shaman of the tribe knew which roots and plants had possitive effects on the condition of sick people, of course he didn't know of anything about chemistry or why they work, but it is the pre-scientific era, so who cares?! Slowly, the mankind became better at using drugs but just as a consequence of a long trial and error proccess. It was not until the scienfic revolution when it started an active and systematic search for new medicines which lead us to the golden-age of medicine in middle of the 20th century when it was a explossion in the number of antibiotics, anticancer or antifungal treatment. It seemeed that in few years we would be able to cure anything. Unfortunately that was not the case, the infections are becoming resistant to the current treatments and cancer treatments are ineffective and unspecific. Unfortunately the drug-discovery process is slow and expensive and since the number of *drugs-to-be-found* is finite it is becoming more and more difficult to identify unknown chemicals making the process more difficult. 

We clearly need new approaches to speed up the process of drug discovery and in this series of post I will show you how to use ML and AI to reach that goal, through several post we will be disecting some *state of the art* ML-driven strategies that you can use to find new drugs and I will give you the knowledge and the tools so you can do it on you own. 

## First approach.
When we create a ML model we are creating and algorithm that can learn to approximate a function *f(x)=y*. In our case the function that we want to approximate is simple *f(drug)=activity* because if we can learn that function we can instantly predict the activity of a drug without testing it in the lab, saving thousands of hours and tons of money in the process. You will see by yourself how you can screen thousands of *unknown* drugs in minutes.

The fist thing that we need to do if we want to approximate the function *f(drug)=activity* is to parameterize the two terms of the function. The parameterization of the *activity* term is easy because we can measure the activity, it can be EC50 which is the concentration at which the drug has the half of the maximum achivable effect, it can be in the form of pKi which is a measurment of the strenght of a drug-protein interaction or it can be in whatever units. It can be continious ranging between two values or discrete, defining two or more classes (no-activity, moderate-activity, high-activity). It doesn't matter as long as you can measure it and it is useful for the inferences you want to make about other unknown drugs. 

But what about the drugs. How do we parameterize them? There are several ways and today I am just going to talk about the simplest way of doing it: using chemical descriptors. 

## Chemical descriptors. 
The chemical descriptors 


 
