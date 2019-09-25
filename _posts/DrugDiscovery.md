---
layout: post
title:  "Computational Drug Discovery (Novice Mode)"
date: '2019-08-25 21:30:00'
---

## Introduction
We humans learned how to use drugs many thousands of years ago. The shaman of the tribe knew which roots and plants had possitive effects on the condition of sick people, of course he didn't know of anything about chemistry or why they work, but it is the pre-scientific era, so who cares?! Slowly, the mankind became better at using drugs but just as a consequence of a long trial and error proccess. It was not until the scienfic revolution when it started an active and systematic search for new medicines which lead us to the golden-age of medicine in middle of the 20th century when it was a explossion in the number of antibiotics, anticancer or antifungal treatment. It seemeed that in few years we would be able to cure anything. Unfortunately that was not the case, the infections are becoming resistant to the current treatments and cancer treatments are ineffective and unspecific. Unfortunately the drug-discovery process is slow and expensive and since the number of *drugs-to-be-found* is finite it is becoming more and more difficult to identify unknown chemicals making the process more difficult. 

We clearly need new approaches to speed up the process of drug discovery and in this series of post I will show you how to use ML and AI to reach that goal, through several post we will be disecting some *state of the art* ML-driven strategies that you can use to find new drugs and I will give you the knowledge and the tools so you can do it on you own. 

## First approach
When we create a ML model we are creating and algorithm that can learn to approximate a function *f(x)=y*. In our case the function that we want to approximate is simple *f(drug)=activity* because if we can learn that function we can instantly predict the activity of a drug without testing it in the lab, saving thousands of hours and tons of money in the process. You will see by yourself how you can screen thousands of *unknown* drugs in minutes.

The fist thing that we need to do if we want to approximate the function *f(drug)=activity* is to parameterize the two terms of the function. The parameterization of the *activity* term is easy because we can measure the activity, it can be EC50 which is the concentration at which the drug has the half of the maximum achivable effect, it can be in the form of pKi which is a measurment of the strenght of a drug-protein interaction or it can be in whatever units. It can be continious ranging between two values or discrete, defining two or more classes (no-activity, moderate-activity, high-activity). It doesn't matter as long as you can measure it and it is useful for the inferences you want to make about other unknown drugs. 

## Drug representation
One of the first things that we need to decide is how we represent the drugs. Drugs can be represented in two ways: we can draw the molecule or we can use the so called [SMILES](https://en.wikipedia.org/wiki/Simplified_molecular-input_line-entry_system). SMILES is an sequence of characters that represent the chemical structure of a molecule without lossing information about it, that means that you can easily interconvert SMILES and drug drawings. 
![SMILES](/images/SMILES.png)
Since most of the information about the drug is retained by the SMILES they are very convinient ways to encode the drug, you can store many of them in very little space. You coud for instance store millions of them in an USB stick.

In latter posts I will show you other interesting properties of the SMILES so they can be used in convolutional neural networks and/or in recursive neural networks. 

Now that we know how to represent the drugs we need to find a way to parameterize them. 
There are several ways and today I am just going to talk about the simplest way of doing it: using chemical descriptors. 

## Chemical descriptors. 
Chemical descriptors are numerical representations that cover different chemical propierties of the drugs. There are hundreds of molecular descriptors that parametrizes different features of the molecules, from the number of atoms or bonds to the solubility in water. By computing the chemical descriptors of a molecule we are predicting the physicochemical, topological, electronical and quantum properties of it and I say prediciting because chemical descriptors of the molecule are calculated algorithmically, that means that some of them such as logP, logD or logS are just predictions. 

Ok. We have found a simple way to parametrize a drug by defining each it as a vector of chemical descriptors that we can use to predicit its bioactivity. 

The hypothesis is simple: drugs with simmilar chemical descriptors will have simmilar activities. Let's test the hypothesis out.

 ### Chemical descriptors calculation.
 First of all we need a dataset to test our hypothesis. In this case I am going to use a dataset of molecules downloaded from the [ZINC15](https://zinc15.docking.org/) database, which is a huge database of compounds, some of them with known properties other with unknown properties. 
 I have downloaded a set of 6102 molecules which are described to be proteases inhibitors. Proteases are enzymes that cut  proteins. While some proteases have important biological roles for the physiology of the cell others are essential for viruses to replicate and proteases inhibitors that espficically inhibit viral proteases are widely used as antivirals to treat HIV or Hepatitis C. You can download the dataset yourself fomr ZINC15 or download it from [here](/images/proteases.smi).  
 
 ![Protinh](/images/Protease-Inhibitors.jpg)
    
Once you have the file we are going to load it into memory, but first let's load the required libraries for the calculations (note that <code>Rcpi</code> needs to be installed through BioConductor and that if you are installing <code>rcdk</code> in linux you will have to install <code>rJava</code> first )

{% highlight r %}
library(Rcpi)
library(rcdk)
library(keras)
library(ggplot2)
{% endhighlight %}

and we load the paths where the training and test data are. 

{% highlight r %}
#Folders
path<-"C:/home/nacho/DrugDiscoveryWorkshop-master/SMILES_Activity.csv"
pathDB<-"/home/nacho/DrugDiscoveryWorkshop-master/ZINCDB/"
{% endhighlight %}

Next, we load the training set and we extract the chemical descriptors with the function <code>extractDrugAIO</code>. You can got and get a cup a cup of coffee because it's going to take a while.

{% highlight r %}
#Loading and descriptors calculation
df<-read.csv(path, sep = " ", header = FALSE)
df<-df[,c(1,5)]
df<-df[-1,]
colnames(df)<-c("SMILES","Affinity")
df$Affinity<-as.numeric(as.character(df$Affinity))
df<-df[complete.cases(df),]
mols <- sapply(as.character(df$SMILES), parse.smiles)
desc <- extractDrugAIO(mols,  silent=FALSE)
{% endhighlight %}

In order to save that time calculating the descriptors if you are cosidering playing around with the script, I recomend you to save the descriptors in a csv file that you can load each time you want to test something.

{% highlight r %}
#Checkpoint
#write.csv(desc, "/home/nacho/StatisticalCodingClub/parameters.csv")
#desc<-read.csv("/home/nacho/StatisticalCodingClub/parameters.csv")
{% endhighlight %}

Now we need to clean up some of the variables 


{% highlight r %}
#Standarize based on previous data
scale.db<-function(df.new, df.old){
  if(ncol(df.new)!=ncol(df.old)) print("Error number of columns!")
  for (i in 1:ncol(df.new)) {
    df.new[,i]<- (df.new[,i]-mean(df.old[,i]))/sd(df.old[,i])
  }
  return(as.matrix(df.new))
}
{% endhighlight %}





{% highlight r %}


{% endhighlight %}


## Sources of images
[SMILES](https://en.wikipedia.org/wiki/Simplified_molecular-input_line-entry_system)
 [Protease inhibitors](https://aidsinfo.nih.gov/understanding-hiv-aids/glossary/603/protease-inhibitor)
