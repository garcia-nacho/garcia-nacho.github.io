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

Now we need to clean up some of the variables. First we remove those that contain any NA value 

{% highlight r %}
#Automatic cleaning
desc$X<-NULL
findNA<-apply(desc, 2, anyNA)
desc<-desc[,which(findNA==FALSE)]
{% endhighlight %}

Next, we clean those variables in which all the values are the same for al drugs. We do it calculating the standard deviation of the different variables, if the sd is zero, that means that all the values of the variable are the same, so we keep only those variables with non-zero sd.

{% highlight r %}
findIdentical<- apply(desc, 2, sd)
desc<-desc[,which(findIdentical!=0)]
{% endhighlight %}

and finaly we manually clean some of the variables 

{% highlight r %}
#Manual cleaning
to.remove<-c("ATSc1",
             "ATSc2",
             "ATSc3",
             "ATSc4",
             "ATSc5",
             "BCUTw.1l",
             "BCUTw.1h",
             "BCUTc.1l",
             "BCUTc.1h",
             "BCUTp.1l",
             "BCUTp.1h",
             "khs.tCH",
             "C2SP1")

`%!in%` = Negate(`%in%`)
desc<-desc[,which(names(desc) %!in% to.remove)]
{% endhighlight %}

Then we standarize the values in the colums (the mean of the variables will be zero and the sd one) and we normalize the values of the activity so they range between zero and one:

{% highlight r %}
x<-scale(desc)
y<-(df$Affinity-min(df$Affinity))/(max(df$Affinity)-min(df$Affinity))
{% endhighlight %}

Now it is time to prepare the training and validation datasets by the drugs into 90% training 10% validation:

{% highlight r %}
#Train/Test
val.ID<-sample(c(1:nrow(desc)),round(0.1*nrow(desc)))

x.val<-x[val.ID,]
y.val<-y[val.ID]
x<-x[-val.ID,]
y<-y[-val.ID]
{% endhighlight %}

We create the model with Keras/TensorFlow...

{% highlight r %}
model<-keras_model_sequential()

model %>% layer_dense(units = 80, activation = "relu",  input_shape = (ncol(x))) %>% 
  layer_dense(units = 40, activation = "relu") %>% 
  layer_dense(units = 10, activation = "relu") %>% 
  layer_dense(units = 1, activation = "linear")
  
{% endhighlight %}

...compile it...
{% highlight r %}
compile(model, optimizer = "adagrad", loss = "mean_absolute_error")
{% endhighlight %}

and train it

{% highlight r %}
history<-model %>% fit(x=x,
                       y=y,
                       validation_data=list(x.val, y.val),
                       callbacks = callback_tensorboard("logs/run_test"),
                       batch_size=5,
                       epochs=10
                        )
plot(history)
{% endhighlight %}

![loss](/images/historydrugs.png)

As you can see, the training was as expected and although it seems that we are not suffering from overfitting I decided to run the training only for 10 epochs.

Let's see how well the model generalizes by trying to predict the activity of the validation set:

{% highlight r %}
#Prediction over validation
y.hat<-predict(model, x.val)

ggplot()+
  geom_jitter(aes(x=y.val, y=y.hat), colour="red", size=2, alpha=0.5)+
  geom_smooth(aes(x=y.val, y=y.hat),method='lm',formula=y~x)+
  ylim(0,1)+
  xlab("Activity")+
  ylab("Predicted Activity")+
  theme_minimal()
{% endhighlight %}

![Yhat](/images/activityprediction.png)

Wooow! Our first model is able to predict the activity of the drugs fairly well. But let's try to optimize it to see if there is a lot of space for improvement.
To try to improve the model we are going to test different hyperparameters. In this case we only modifying the number neurons and some activation functions while keeping the main architecture intact. In following posts I will show you how to modidify the architecture of the model as another hyperparameter.

Briefly, we iterate over 200 models saving the hyperparameters in a dataframe and we test the mean square error for each model.

{% highlight r %}
#Optimization
reset_states(model)
rounds<-200
models<-as.data.frame(matrix(data = NA, ncol =9 ,nrow = rounds))
colnames(models)<-c("UnitsL1", "UnitsL2", "UnitsL3", "act", "L1.norm", "L2.norm", "L1.norm.Rate", "L2.norm.Rate", "MSE")
pb<-txtProgressBar(min=1, max = rounds, initial = 1)

for (i in 1:rounds) {

  #Hyperparams
  unitsL1<-round(runif(1, min=20, max = 100))
  unitsL2<-round(runif(1, min=10, max = 60))
  unitsL3<-round(runif(1, min=2, max = 20))
  actlist<-c("tanh","sigmoid","relu","elu","selu")
  act<-actlist[round(runif(1,min = 1, max=5))]
  L1.norm<-round(runif(1,min = 0, max = 1))
  L2.norm<-round(runif(1,min = 0, max = 1))
  L1.norm.Rate<-runif(1, min = 0.1, max = 0.5)
  L2.norm.Rate<-runif(1, min = 0.1, max = 0.5)
  ###
  
  setTxtProgressBar(pb, i)
  model.search<-keras_model_sequential()
  
  model.search %>% layer_dense(units = unitsL1, activation = act, input_shape = (ncol(x)))
  if(L1.norm==1)  model.search %>% layer_dropout(rate=L1.norm.Rate)
  model.search %>% layer_dense(units = unitsL2, activation = act)
  if(L2.norm==1)  model.search %>% layer_dropout(rate=L2.norm.Rate)   
  model.search %>% layer_dense(units = unitsL3, activation = act) %>% 
  layer_dense(units = 1, activation = "sigmoid")
  
  compile(model.search, optimizer = "adagrad", loss = "mean_squared_error")
  model.search %>% fit(x=x,
                         y=y,
                         verbose=0,
                         batch_size=10,
                         epochs=10
  )
  
  y.hat<-predict(model.search, x.val)
  MSE<-(sum((y.val-y.hat)^2))/length(y.val)
  
  #FIlling dataframe
  models$MSE[i]<-MSE
  models$act[i]<-act
  models$UnitsL1[i]<-unitsL1
  models$UnitsL2[i]<-unitsL2
  models$UnitsL3[i]<-unitsL3
  models$L1.norm[i]<-L1.norm
  models$L2.norm[i]<-L2.norm
  models$L1.norm.Rate[i]<-L1.norm.Rate
  models$L2.norm.Rate[i]<-L2.norm.Rate
  reset_states(model.search)
  }
  {% endhighlight %}

Next we select the best possible model and we train on it to use it to predict the activity of the test dataset:

{% highlight r %}
#Validation best performer
models<-models[complete.cases(models),]
unitsL1<- models$UnitsL1[models$MSE==min(models$MSE)]
unitsL2<- models$UnitsL2[models$MSE==min(models$MSE)]
unitsL3<- models$UnitsL3[models$MSE==min(models$MSE)]

act<-models$act[models$MSE==min(models$MSE)]
L1.norm<-models$L1.norm[models$MSE==min(models$MSE)]
L2.norm<-models$L2.norm[models$MSE==min(models$MSE)]
L1.norm.Rate<-models$L1.norm.Rate[models$MSE==min(models$MSE)]
L2.norm.Rate<-models$L2.norm.Rate[models$MSE==min(models$MSE)]

model.best<-keras_model_sequential()

model.best %>% layer_dense(units = unitsL1, activation = act, input_shape = (ncol(x)))
if(L1.norm==1)  model.best %>% layer_dropout(rate=L1.norm.Rate)
model.best %>% layer_dense(units = unitsL2, activation = act)
if(L2.norm==1)  model.best %>% layer_dropout(rate=L2.norm.Rate)   
model.best %>% layer_dense(units = unitsL3, activation = act) %>% 
  layer_dense(units = 1, activation = act)

compile(model.best, optimizer = "adam", loss = "mean_squared_error")
model.best %>% fit(x=x,
                     y=y,
                     
                     batch_size=10,
                     epochs=10
)

y.hat<-predict(model.best, x.val)

ggplot()+
  geom_jitter(aes(x=y.val, y=y.hat), colour="red", size=2, alpha=0.5)+
  geom_smooth(aes(x=y.val, y=y.hat),method='lm',formula=y~x)+
  ylim(0,1)+
  xlab("Activity")+
  ylab("Predicted Activity")+
  theme_minimal()
{% endhighlight %}

We save the model so we can reuse it or share it. Download my trained model [here.](/images/modelNNet.h5) 

{% highlight r %}
save_model_hdf5(model.best, "/home/nacho/Drugs/BestModelNN.h5")
{% endhighlight %}

Now we are ready to find new protease-inhibitors among large datasets. 
Since the next chunk of code contains a big while loop that I don't want to cut, I will explain how it works here:

First, we define the folder in which the SMILES are. 
Next, we check that the files cotaining the SMILES have not been processed yet. This part is very useful because it allows you to add more files to the folder while the script is running so it keeps going iterating over the new files.
Then, in order to run of memory when we process the SMILES we load them in batches of 10000. 


#Screening
db.completed<-vector()
db<-list.files(pathDB)
db.toprocess<-setdiff(db,db.completed)
if(length(db.toprocess>0)) continue=TRUE







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
