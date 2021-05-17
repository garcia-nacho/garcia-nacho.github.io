---
layout: post
title:  "Finding anomalies in other stars"
date: '2019-02-18 19:25:00'
---

### Introduction.
**"Are we alone in the universe?"**.  
When it comes to answering one of the most relevant questions that humanity has still open, we have to admit that we don't have a clue. 
The first historical references to the question are found in ancient Greece. Greek and Roman philosophers reasoned that since the universe seemed to be so vast it would be reasonable to think that other "tribes" would be living somewhere out there. Later on, religious thinking displaced the logical reasoning and humanity became special just because it was God's will for the humans to be special. It was not until the Renascent when others started to ask the same question again: *"If the Earth is no the center of the universe and there are other planets and stars, why should be life on Earth special?"*   

If you think for a second on the implications that answering that question would have, you would realize how crucial it is for us, humans, find the answer. It doesn't really matter the answer itself, just knowing with 100% certainty that there is or that there is not life somewhere else in the universe would change us as a whole. 

Unfortunately, neither ancient greeks or guys like Copernicus or Huygens had the technology to find it out and indeed we still don't know if we can answer that question today, but today in this post, I will show you some of the current approaches that we, humans, are using to try to answer that question. 

{: style="text-align: justify"}
<!--more-->

### The Fermi paradox and the Drake equation
In 1961 Frank Drake tried to shed some light into the mystery by turning the philosophical question into a parametric hypothesis. He estimated numerically the probability of another intelligent civilization living in our galaxy.  For the first time in history, someone had answered the question from a numerical point of view.  According to Drake, we are sharing our galaxy with another 10 intelligent civilizations with the ability to communicate. Is that too much? is that too little? Well... for sure we would not be able to detect our own presence from a distance 10K light-years (3,1kpc) which is approximately 1/10 of the galactic size. Indeed we would not be able to detect even the presence of Earth since the furthest exoplantet detected and confirmed it is around 6.5K light-years:

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

Of course, some of Drake equation's constrains are purely theoretical; for instance, we don't know how easy it is for a form of life to exist or to become intelligent. There is a recent study in which the authors develop a theoretical Bayesian framework to estimate some of Drake's parameters. In this paper they argue that considering how early life arrived to Earth and how late this life became intelligent[Ref](https://www.pnas.org/content/117/22/11995), adquisition of intelligence might not be an easy leap to make; however, the universe is vast, so big that even events with very low probality are likely to happen somewhere and this is where the Fermi paradox comes in: If it is so likely that there is intelligent life somewhere in the observable universe, why haven't we heard from them??!! 
There are so many possible answers to try to solve the paradox that it is impossible to mention them all, and still it is a paradox because none of the arguments is strong enough to solve it. Maybe aliens are not interested in communicate, maybe they are too advanced for us to understand them, maybe there are "technological filters" like nuclear wars, climate changes, etc... wipping out civilizations once they discover these technological stages.    
Again, we don't know.
{: style="text-align: justify"}

### The SETI/Breakthrough programs 
Since we haven't been able to disprove any hypothesis regarding extraterrestrial intelligence so far, we can only keep scanning the sky looking for a signature of an intelligent civilization and this is what the SETI does. Through different initiatives, SETI scans most of the electromagenetic spectrum (from radio waves to light sources) trying to find unnatural signals. However the amount of data gathered so far is insanely big, that big that a comprenhensive analysis using "humans-informed" analyses would take a lot of years. We need machines to do the job.
![MatrixQ](/images/matrxquote.jpeg)

SETI, through the Breakthrough program, has released a masive dataset containing data from the closests 100000 stars. The data comes from [The Parkes Observatory](https://en.wikipedia.org/wiki/Parkes_Observatory), [The Green Bank Telescope](https://greenbankobservatory.org/science/telescopes/gbt/) and the [Automated Planted Finder(Lick Observatory)](https://en.wikipedia.org/wiki/Automated_Planet_Finder) and you can download everything from [here](https://breakthroughinitiatives.org/opendatasearch).
The idea behind releasing such a dataset is to make it available for the community, so everyone can explore it and this what I will do in this post:to establish a computational framework so everyone can explore the dataset. 

### Navigating the data. 
In this post I will just focus on the APF dataset because of it's simplicity. The APF dataset is a set of images in [FITS](https://en.wikipedia.org/wiki/FITS) format. Each image corresponds to the observation of one star at one particular timepoint. The "images" are indeed spectrograms which are generated by a telescope coupled with an [echelle spectrograph](https://en.wikipedia.org/wiki/Echelle_grating) that splits the light coming from an star into its different wavelenghts. This allows us to measure the number of photons of each wavelength coming from that particular start:
Here is an scheeme of an echelle spectograph:

![Echelle](/images/micromachines-10-00037-g002.png)  [Source](https://www.mdpi.com/2072-666X/10/1/37/htm)

The raw images that we obtain from the echelle telescope look like this:

![Echelle image](/images/echelle1.png)

The image represents the diffaction of the light of a star. THe wavelenght increases from up to low and from left to right, so the photons with the lower wavelength (red) hit the top left corner and in the lower left corner shows the amount of photons with the highest wavelength (blue). If we assing a colour to each region we would have something like this:

![ColouredEchelle](/images/echelle-spectrum.jpg)
[source](https://blogs.maryville.edu/aas/echelle-spectrum/)    

As you can see there are a couple of features in the image, the shadows and the "tracks". This is more obvious if we zoom in into the image:

![Echelle zoom](/images/echelle%20zoom.png) 

This is basically an artifact of the technique, not all area of the image is covered by the diffracted light, that means that we are only interested in the tracks of light which are called orders. We therefore need a way to extract the useful information (the tracks) from the image, this process is called compression (basically because the resulting image with the important information is much smaller in size), but to do it we need to know where the tracks lay in the image. 
The way to do it is a bit convoluted: We need to find the functions that draw a line going though all the pixels of every track. 
As you can see in the image, the tacks are curved and although this is totally normal in echelle spectra, it is a bit problematic for us because the functions tha draw such a curves are 4rd order polynomial functions like this one:   
$$y=a+b\cdot x+c\cdot x^{2}+d\cdot x^{3}+e\cdot x^{4}$$   

so we *only* need a way to find a, b, c and d for each order (and in total we have 79. I guess that different processing softwares have their own but I had to develop my own algorithm to estimate the 5 parameters.

My algorithm basically find the orders by identifying the tracks in the central region of the image, which is where all the tracks are better defined and then it goes track by track finding the parameters that satisfy the following two conditions: 
  
1. The curve must go through the point of the track identified in the central retgion.
2. The mean of the pixels of the that the curve overlay must be maximize (meaning that we are avoiding the shades between tracks)

So let's start coding it: 

Before doing anything we need to load the FITS files and this is done in R using the FITSio library (we also load the rest of the libraries that we are going to use later on)

{% highlight r %}
library(FITSio)
library(doParallel)
library(parallel)
library(foreach)
library(data.table)
library(ggplot2)
library(ggpubr)
library(ggrepel)
library(RCurl)

df<-readFITS("/home/nacho/SETI/ucb-bek230.fits")
df.img<-df$imDat

{% endhighlight %}

You can download the ucb-beck230.fits file from [here](/images/ucb-bek230.fits). Now the image is stored in the df.img matrix with dimensions [2080,4608]. Although you can plot the image I don't recomend it, the rendering it is very slow and it is going to be difficult to see anythin. Anyway if you insist this is the best I have obtained by tweaking the zlim of the image function:

<code>image(df.img, zlim = c(median(df.img)/2, median(df.img)*2))</code>

as you can see the rendering is so inefficient that it creates the horizontal white stripes that you can see on the image:
![EchelleR](/images/Rplot27.png)   
Next we need to remove the background of the image and we do it by substracting the value corresponding to the first 30 rows of the image, that never have a track:

{% highlight r %}
baseline<-median(df.img[1:30,])
df.img<-df.img-baseline
{% endhighlight %}
 
Then, we look for the 79 peaks of photons corresponding to the orders at the center of the image:

{% highlight r %}
#Identification of N peaks 
df.img<-df.img-baseline
anchor.band<-round(ncol(df.img)/2)
track.width <- 10
track.n<-79
central.band<- df.img[,anchor.band]
#Remove cosmic ray
central.band[which(central.band>2500)]<-0

anchor.point <- vector()
for (i in 1:track.n) {
  anchor.point.dummy <- which(central.band==max(central.band))[1]
  central.band[(anchor.point.dummy-track.width): (anchor.point.dummy+track.width)]<-0
  anchor.point<-c(anchor.point, anchor.point.dummy)
}
anchor.point<-unique(anchor.point)
{% endhighlight %}

In this algorithm, the peaks are stored in the anchor point vector.

This is how the order look like when they cross the central part of the image:
![AnchorPoints](/images/anchorpoints.png)
and his is how the algorithm finds the peaks:
![AnchorPoints2](/images/anchorpoints2.png)

Now that we have identified the orders in the middle part of the spectrogram, we need to identify the tracks. Note that the algorithm has some problems identifying lowest and highest orders (the signal in those wavelengths is so weak that the algorithms fails detecting them over the background noise.

As I mentioned above we want to find those 4rd oder functions that overlap the tracks, since the function to be maximized is not differentiable we will find the parameters by doing a random search. Parameters *b, c, d,* and *e* will take values on a small range so it will be easier for the model to find the optimal solution after few iterations (10000). Parameter *a* will take the only possible value that make the curve go through the pixel identified to belong to the order at the cental region. 
Since the algorithm has to find the parameters for each one of the 79 orders identifed it offers a good opportunity to paralelize the code, so each processor gets the task of finding the parameters for one order and here is the code that does it:

{% highlight r %}
fitter <- function(x, a, b, c, d, e){
  y<-a + b*x + c*x^2 + d*x^3+ e*x^4 
  return(round(y))
}

iterations<-10000

cores<-detectCores()-1
cluster.cores<-makeCluster(cores)
registerDoParallel(cluster.cores)

output<-foreach(i=1:length(anchor.point), .verbose = TRUE) %dopar%{
  #Function to call equation

#for (i in 1:length(anchor.point)) {
  
  x<-round(ncol(df.img)/2)
  y<-anchor.point[i] 
  for (k in 1:iterations) {

  #Parameter generation
  b <-runif(1, min = -0.08, max= 0)
  c <-runif(1, min = 0, max= 1.6e-05)
  d <-runif(1, min = -1.1e-09, max= 1.5e-09)
  e<- runif(1, min=-1e-13,max= 2e-13)
  a<- y - b*x - c*x^2 - d*x^3 - e*x^4 
  
  #call function
  row.ids<-fitter(c(1:ncol(df.img)),a,b,c,d,e)
  
  row.ids<-row.ids[row.ids>0]
  row.ids<-row.ids[row.ids<nrow(df.img)]
  
  if(length(row.ids)>100){
  indexes<-vector()
  for (j in 1:length(row.ids)) indexes[j]<-((2080)*(j-1))+row.ids[j]
  
  #Create evaluation metric
  score<-mean(df.img[indexes])
  
  if(!exists("best.score")) {
    best.score<-score
    best.params <- c(a,b,c,d,e,i)
    }
    
  if(score>best.score){
    best.score<-score
    best.params<-c(a,b,c,d,e,i)
    } }
  }
  return(best.params)
  rm(best.params)
  rm(best.score)
}

stopCluster(cluster.cores)

params.opt<-output[[1]] 
for (i in 2:length(output)) {
  params.opt<-rbind(params.opt, output[[i]])
  
}
{% endhighlight %}

Now that we have found where the orders lay in the image, we need to extract them. To do that, we average the value of the width of the track (approximately five pixels) 


{% highlight r %}

{% endhighlight %}

Unfortunately, the random nature of the paarameters finding the orders would be uncomparable to each other unless we aligned them using some common spectral features (eg. The H-band due to the hydrogen and/or iodine bands used to calibrate echelle spectographs). However, that goes beyond the scope of the article because, luckily for us we don't need to do the entire process for each spectrogram. We can just donwload the compressed spectrogram from the SETI webpage.  

### Scrapping SETI 
Now that we have a plan A (to download the compressed spectrogram) and a plan B (to compress the raw spectrogram when it is needed) we can procceed to download the data from the breakthrough database. To do that we would use an index file of the database, so we can get only the files that are interesting for us. You can download the file [here](/images/apf_log.txt) and this is the code to get the information:

{% highlight r %}
files<-read.csv("/home/nacho/SETI/apf_log.txt", sep = " ", header = FALSE)
files<-files[-1,]
files<-files[,1:8]
files$reduced<-gsub(".*\\/r", "TRUE", files$V1)
files$reduced<-gsub("TRUE.*", "TRUE", files$reduced)
files<-files[files$reduced=="TRUE", ]

stars<- unique(as.character(files$V3))
stars.n<-vector()
for (i in 1:length(stars)) {
  stars.n[i]<-nrow(files[files$V3==as.character(stars[i]),])
  
}
stars<-stars[order(stars.n, decreasing = TRUE)]
stars.n<-stars.n[order(stars.n, decreasing = TRUE)]
{% endhighlight %}

This code gets the information about the stars that are in the database and the number of observations for each one of them. 
From there you could extract the information about your favourite star. In this example I am going to get the information about the Tabby star (aka. 8462852), to do that I just need to find and download the files in the dtabase that correspond to observations of Tabby:

{% highlight r %}
tabby <- files[grep("8462852", files$V3),]

tabby$link<-gsub("gs:\\/", "https:\\/\\/storage.googleapis.com", tabby$V1)
tabby$filename<-gsub(".*\\/","",tabby$V1)

for (i in 1:nrow(tabby)) {
  download.file(tabby$link[i], paste("/home/nacho/SETI/Tabby/",tabby$filename[i],sep = ""))
}

{% endhighlight %}

I am sure that you have already heard of Tabby, although you might not recognize it by the name. Tabby is a star that became very famous a few years ago when it was found that it had abnormal reductions of the light intensity that were not consistent with any stellar event or the presence of planets orbiting it [ref](). Based on this unusual observations XXXX proposed that an hypotheis that fits with the data would be the presence of a Dyson Sphere. Which is an hypothetical alien megastructure developed by very advanced civilization to extract the energy from a star. There have been propossed several types of Dyson Speres, from a rigid gigant sphere to a swarm of smaller satelites devoted to get the star energy.

Although the mystery is still unsolved, no radio emisions from Tabby have been detected and the strongest hypothesis nowadays is that the  fluctuations could be originated by a cloud of asteroids/comets/dust there is a strong interest on this star by the media, so let's surf the vawe of trendiness and explore Tabby.

Now that we have all the compressed spectra from Tabby in the same folder we can load them into a 3D array, although there are different approaches to deal with time-series in R in this case I fnd this one more convinient because it just treates the spectra as if they were images and the 3rd dimension would be the temporal dimension, it is like doing a time-lapse with the spectra.

{% highlight r %}
### Load files directory Taby--------------
files<-list.files("/home/nacho/SETI/Tabby/")

image.array<-array(data = NA, dim = c(length(files), 4608, 79))
date<-vector()
for (i in 1:length(files)) {
  dummy<-readFITS(paste("/home/nacho/SETI/Tabby/", files[i], sep = ""))
  date[i]<-dummy$header[25]
  image.array[i,,]<-dummy$imDat
}

wl<-readFITS("/home/nacho/SETI/apf_wav.fits")
wl<-wl$imDat

{% endhighlight %}

You have probably noticed that the scripts uses an additional file [apf_wav.fits](/images/apf_wav.fits). That file can be used to extract the information regarding the correspondance between the wavelenght and the possitions of the spectra. 

Next, we normalize the data

{% highlight r %}
#Normalization
photons_norm<-vector()
for (timeevents in 1:dim(image.array)[1]) {
  photons_norm[timeevents]<-sum(image.array[timeevents,,])
  
}
photons_norm<-photons_norm/min(photons_norm)

for (timeevents in 1:dim(image.array)[1]) {
  image.array[timeevents,,]<-image.array[timeevents,,,drop=FALSE]/photons_norm[timeevents]
  
}

{% endhighlight %}


https://www.pnas.org/content/115/42/E9755




{% highlight r %}
linear<-lm(log(df$Infected-1)~df$Day)
start <- list(a=exp(coef(linear)[1]), b=coef(linear)[2])
{% endhighlight %}
