library(FITSio)
library(doParallel)
library(parallel)
library(foreach)
library(data.table)
library(ggplot2)
library(ggpubr)
library(ggrepel)
library(RCurl)

rotate <- function(x) t(apply(x, 2, rev))
column.summ<-function(x) apply(x, 2, sum)

df<-readFITS("/home/nacho/SETI/ucb-bek230.fits")

df.img<-df$imDat

image(df.img, zlim = c(median(df.img)/1.5, median(df.img)*2))

dim(df.img)

df.img.t<-rotate(rotate(df.img))

image(t(df.img.t[1000:1100, 2199:2300]),zlim = c(median(df.img.t), median(df.img.t)*15))

mean.df.col<-apply(df.img, 2, mean)
sd.df.col<-apply(df.img, 2, sd)


plot(column.summ(df.img.t[1684:1689, 2199:2799]), type = "l")

baseline<-median(df.img[1:30,])

df.img.t<-df.img.t-baseline

#H-band
plot(column.summ(df.img.t[1491:1506,1510:2500]), type = "l")
plot(column.summ(df.img.t[1491:1506,1500:2500]), type = "l")

#
image(t(df.img.t[1333:1348, 1200:2200]),zlim = c(median(df.img.t[1333:1348, 1200:2200]), median(df.img.t[1333:1348, 1200:2200])*15))

plot(column.summ(df.img.t[1333:1348, 1200:2200]), type = "l")

coef<-read.csv("/home/nacho/SETI/order_coefficients copy.txt", sep = " ", header = FALSE)
coef<- apply(coef, 1,na.exclude ) 
coef<- apply(coef, 1,na.exclude )

fopol.fit <- function(a,b,c,d,e,x){
  y<-a + b*x + c*x^2 + d*x^3 + e*x^4
    return(y)
}


#Reduction algorithm 

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
anchor.point<-anchor.point[order(anchor.point)]

plot(df.img[,anchor.band], type = "l", ylab = "Intensity")
points( anchor.point, rep(40, track.n), col="red")

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
  
  
#write.csv(params.opt,"/home/nacho/SETI/optimalparams.csv")

params.opt<- read.csv("/home/nacho/SETI/optimalparams.csv")
plot(fitter(c(1:ncol(df.img)),params.opt[1,1],params.opt[1,2],params.opt[1,3],params.opt[1,4],params.opt[1,5]), 
     type = "l",  ylab = "Y", xlab = "X")
for (i in 2:60) {
  lines(fitter(c(1:ncol(df.img)),params.opt[i,1],params.opt[i,2],params.opt[i,3],params.opt[i,4],params.opt[i,5]))
} 

#Order extraction
for (i in 1:length(anchor.point)) {
  df.coord<-  as.data.frame(fitter(c(1:ncol(df.img)),a,b,c,d,e))
  colnames(df.coord)<-"Y"
  df.coord$X<-c(1:ncol(df.img))  
  dummy.row<-vector()
  
  for (j in 1:nrow(df.coord)) {
    dummy.row[j]<-mean(df.img[c((df.coord$Y[j]-2):(df.coord$Y[j]+2)),df.coord$X[j]])
  }
  
}

######################



files<-read.csv("/home/nacho/SETI/apf_log.txt", sep = " ", header = FALSE)
files<-files[-1,]
files<-files[,1:8]
files$reduced<-gsub(".*\\/r", "TRUE", files$V1)
files$reduced<-gsub("TRUE.*", "TRUE", files$reduced)
files<-files[files$reduced=="TRUE", ]
tabby <- files[grep("8462852", files$V3),]

tabby$link<-gsub("gs:\\/", "https:\\/\\/storage.googleapis.com", tabby$V1)
tabby$filename<-gsub(".*\\/","",tabby$V1)

for (i in 1:nrow(tabby)) {
  download.file(tabby$link[i], paste("/home/nacho/SETI/Tabby/",tabby$filename[i],sep = ""))
}

stars<- unique(as.character(files$V3))
stars.n<-vector()
for (i in 1:length(stars)) {
  stars.n[i]<-nrow(files[files$V3==as.character(stars[i]),])
  
}
stars<-stars[order(stars.n, decreasing = TRUE)]
stars.n<-stars.n[order(stars.n, decreasing = TRUE)]



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

#Normalization
photons_norm<-vector()
for (timeevents in 1:dim(image.array)[1]) {
  photons_norm[timeevents]<-sum(image.array[timeevents,,])
  
}
photons_norm<-photons_norm/min(photons_norm)

for (timeevents in 1:dim(image.array)[1]) {
  image.array[timeevents,,]<-image.array[timeevents,,,drop=FALSE]/photons_norm[timeevents]
  
}


#Spike finder
mean.array<-matrix(data = NA, nrow = dim(image.array)[2], ncol = dim(image.array)[3])
sd.array<-matrix(data = NA, nrow = dim(image.array)[2], ncol = dim(image.array)[3])
median.array<-matrix(data = NA, nrow = dim(image.array)[2], ncol = dim(image.array)[3])

for (i in 1:dim(mean.array)[2]) {
  mean.array[,i]<-apply(image.array[,,i], 2, mean)  
  sd.array[,i]<-apply(image.array[,,i], 2, sd)
  median.array[,i]<-apply(image.array[,,i], 2, median)  
}

plot(wl[1500:2500,54], image.array[1,1500:2500,54], type = "l", xlab = "Wavelenght (Å)", ylab = "Photon count", main = "Time = 0", ylim = c(40,130))
plot(wl[1500:2500,54], mean.array[1500:2500,54], type = "l", xlab = "Wavelenght (Å)", ylab = "Mean(Photon count)", main = "Mean", ylim = c(40,130))
plot(wl[1500:2500,54], median.array[1500:2500,54], type = "l", xlab = "Wavelenght (Å)", ylab = "Median(Photon count)", main = "Median", ylim = c(40,130))
plot(wl[1500:2500,54], sd.array[1500:2500,54], type = "l", xlab = "Wavelenght (Å)", ylab = "SD(Photon count)", main = "SD")


Hband<-wl[1500+which(median.array[1500:2500,54]==min(median.array[1500:2500,54])),54]

image(median.array)

date<-gsub(".*=..", "",date)
date<-gsub(". / UT.*", "",date)
date<-as.POSIXct(date,format="%Y-%m-%dT%H:%M:%S")
date.julian<-julian.POSIXt(date)
date.julian<-(date.julian-min(date.julian))
date.julian<-as.numeric(date.julian)


photons.max<-image.array[,which(median.array==max(median.array), arr.ind = TRUE)[1],which(median.array==max(median.array), arr.ind = TRUE)[2]]
plot(date.julian, photons.max)

image(median.array)

image(median.array[,45:47])



image(median.array[,40:43])



#Anomaly 1 32 
image(median.array[3600:3650,40:42])
plot(wl[3600:3650,41], image.array[1,3600:3650,41], type = "l", ylim = c(0,5000), ylab = "Photons", xlab = "Wavelength (Å)", main = "Peak1")
for (i in 2:dim(image.array)[1]) {
  lines(wl[3600:3650,41], image.array[i,3600:3650,41])  
}

plot((image.array[,3632,41]-min(image.array[,3632,41]))/(max(image.array[,3632,41])-min(image.array[,3632,41])),type="l", ylab = "Normalized photon count")
lines((image.array[,1599,46]-min(image.array[,1599,46]))/(max(image.array[,1599,46])-min(image.array[,1599,46])),type="l", col="red")



wl[3632,41]

#630 nm peak2
image(median.array[2000:2300,50:52])
plot(wl[2100:2200,51], image.array[1,2100:2200,51], type = "l", ylim = c(0,2000), ylab = "Photons", xlab = "Wavelength (Å)", main = "Peak2")
for (i in 2:dim(image.array)[1]) {
  lines(wl[2100:2200,51], image.array[i,2100:2200,51])  
}

#630 nm peak3
image(median.array[270:310,41:43])
plot(wl[270:310,42], image.array[1,270:310,42], type = "l", ylim = c(0,2000), ylab = "Photons", xlab = "Wavelength (Å)", main = "Peak3")
for (i in 2:dim(image.array)[1]) {
  lines(wl[270:310,42], image.array[i,270:310,42])  
}


#630 nm peak4
image(median.array[3950:4058,74:76])
plot(wl[3950:4058,75],median.array[3950:4058,75], type = "l")
plot(wl[3950:4058,75], image.array[1,3950:4058,75], type = "l", ylim = c(0,1000), ylab = "Photons", xlab = "Wavelength (Å)", main = "Peak4")
for (i in 2:dim(image.array)[1]) {
  lines(wl[3950:4058,75], image.array[i,3950:4058,75])  
}


#Sodium
image(median.array[1500:2000,45:47])
plot(median.array[1500:2000,46], type = "l")
plot(wl[1500:2000,46], image.array[1,1500:2000,46], type = "l", ylim = c(0,10000), ylab = "Photons", xlab = "Wavelength (Å)")

for (i in 2:dim(image.array)[1]) {
  lines(wl[1500:2000,46], image.array[i,1500:2000,46])  
}

wl[1599,46]


plot(image.array[,1599,46],type="l", ylab = "Photon count")
lines(image.array[,1874,46], col="red")




#Correlation Green/Red +57
plot(wl[2150:2160,51], image.array[1,2150:2160,51], type = "l", ylim = c(0,11500), ylab = "Photons", xlab = "Wavelength (Å)")
for (i in 2:dim(image.array)[1]) {
  lines(wl[2150:2160,51], image.array[i,2150:2160,51])  
}

green<-image.array[,3631,41]
red<-image.array[,2157,51]


plot(green, red, xlab = "Green", ylab = "Red", xlim = c(0,1000), ylim = c(0,1000))


total.photons<-apply(image.array, 1, sum)
plot(date.julian, total.photons)


### Substract median to find out noise META------------
noise.array<-image.array
total.noise<-vector()
for (i in 1:dim(image.array)[1]) {
noise.array[i,,] <- image.array[i,,]-median.array
total.noise[i]<-sum(noise.array[i,,])
}
plot(total.noise)

plot( noise.array[1,,], type = "l", ylim = c(0,1000), ylab = "Photons", xlab = "Wavelength (Å)")
for (i in 2:dim(image.array)[1]) {
  lines(noise.array[i,,51])  
}

c.ray.count<-vector()
for (i in 1:dim(image.array)[1]) {
  c.ray.count[i]<-length(which(as.numeric(noise.array[i,,]) > 100))
}
plot(c.ray.count, type = "l")
boxplot(c.ray.count)

#Scan segments of wl for the intensity over time 
segment.size<-100
spec.size <- max(wl)-min(wl)
image.array.fit<-image.array[,1:nrow(wl),]
image.array.fit[image.array.fit<0]<-0
rm(output.segment)
#For loop over time
for (i in 1:dim(image.array)[1]) {
  dummy.array<-image.array.fit[i,,]
segment.array<-data.frame(matrix(NA, ncol = 4, nrow = round(spec.size/segment.size)))
colnames(segment.array)<-c("wl_start", "wl_end", "median","time")
segment.array$wl_start[1]<-min(wl)
segment.array$wl_end[1]<-segment.array$wl_start[1]+segment.size
segment.array$median[1]<- median(dummy.array[ which(wl>=segment.array$wl_start[1] & wl<segment.array$wl_end[1] )] )
segment.array$time<-date.julian[i]

for (i in 2:nrow(segment.array)) {
segment.array$wl_start[i]<-segment.array$wl_end[i-1]  
segment.array$wl_end[i]<-segment.array$wl_start[i]+segment.size
segment.array$median[i]<- median(dummy.array[ which(wl>=segment.array$wl_start[i] & wl<segment.array$wl_end[i])] )
}
if(!exists("output.segment")){
  output.segment<-segment.array
}else{
  output.segment<-rbind(output.segment, segment.array)
}
  
}

ggplot(output.segment)+
  geom_line(aes(wl_start, median))+
  facet_wrap(~round(time,3))+
  theme_minimal()


#Normalization
photons_norm<-vector()
for (timeevents in 1:dim(image.array)[1]) {
  photons_norm[timeevents]<-sum(image.array[timeevents,,])
  
}
photons_norm<-photons_norm/min(photons_norm)

for (timeevents in 1:dim(image.array)[1]) {
  image.array[timeevents,,]<-image.array[timeevents,,,drop=FALSE]/photons_norm[timeevents]
  
}


#Peak 8k 9k

plot(image.array [1, 700:1472, 67], type = "l", ylim = c(0,5000))
for (i in 1:dim(image.array)[1]) {
  lines(image.array [i, 700:1472, 67])
}

#Open other images HIP22449
HIP22449 <- files[grep("HIP22449", files$V3),]

HIP22449$link<-gsub("gs:\\/", "https:\\/\\/storage.googleapis.com", HIP22449$V1)
HIP22449$filename<-gsub(".*\\/","",HIP22449$V1)

for (i in 1:nrow(HIP22449)) {
  download.file(HIP22449$link[i], paste("/home/nacho/SETI/HIP22449/",HIP22449$filename[i],sep = ""))
}


###Load files directory
files<-list.files("/home/nacho/SETI/HIP22449/")

image.array2<-array(data = NA, dim = c(length(files), 4608, 79))
date2<-vector()
for (i in 1:length(files)) {
  dummy<-readFITS(paste("/home/nacho/SETI/HIP22449/", files[i], sep = ""))
  date2[i]<-dummy$header[25]
  image.array2[i,,]<-dummy$imDat
}


plot(wl[2100:2200,51], image.array2[1,2100:2200,51], type = "l", ylim = c(0,110000),  ylab = "Photons", xlab = "Wavelength (Å)")
for (i in 2:dim(image.array2)[1]) {
  lines(wl[2100:2200,51], image.array2[i,2100:2200,51])  
}

#Find maximun peak
peak.position.date <- which(photons.max==max(photons.max))
julian.peak<-as.numeric(julian.POSIXt(date[peak.position.date])) 

files.t<-read.csv("/home/nacho/SETI/apf_log.txt", sep = " ", header = FALSE)
files.t<-files.t[-1,]
files.t<-files.t[,1:8]
files.t$reduced<-gsub(".*\\/r", "TRUE", files.t$V1)
files.t$reduced<-gsub("TRUE.*", "TRUE", files.t$reduced)
files.t<-files.t[files.t$reduced=="TRUE", ]

files.t$date<-as.POSIXct(files.t$V8,format="%Y-%m-%dT%H:%M:%S")
files.t$date<-julian.POSIXt(files.t$date)
files.t$date<-files.t$date-julian.peak

#Remove tabby
files.t <- files.t[-grep("8462852", files.t$V3),]

#Range 4 days
files.t <- files.t[abs(files.t$date)<2,]
files.t$link<-gsub("gs:\\/", "https:\\/\\/storage.googleapis.com", files.t$V1)
files.t$filename<-gsub(".*\\/","",files.t$V1)

for (i in 1:nrow(files.t)) {
  download.file(files.t$link[i], paste("/home/nacho/SETI/4days/",files.t$filename[i],sep = ""))
}

###Load files directory
files<-list.files("/home/nacho/SETI/4days/")

image.array4days<-array(data = NA, dim = c(length(files), 4608, 79))
date4days<-vector()
for (i in 1:length(files)) {
  dummy<-readFITS(paste("/home/nacho/SETI/4days/", files[i], sep = ""))
  date4days[i]<-dummy$header[25]
  image.array4days[i,,]<-dummy$imDat
}


plot(wl[3600:3650,41], image.array4days[1,3600:3650,41], type = "l", ylim = c(0,11500), ylab = "Photons", xlab = "Wavelength (Å)")
for (i in 2:dim(image.array4days)[1]) {
  lines(wl[3600:3650,41], image.array4days[i,3600:3650,41])  
}
lines(wl[3600:3650,41], image.array[1,3600:3650,41], col="red")
for (i in 2:dim(image.array)[1]) {
  lines(wl[3600:3650,41], image.array[i,3600:3650,41],col="red")  
}
### Green line solved: Teluric

##Get a list of anomalies
## The median is two times the median of the surrounding positions
dim(image.array)
plot(as.vector(median.array[1:dim(wl)[1],]), type="l")
median.array<-median.array[-nrow(median.array),]
anomalies<-vector()

plot(as.vector(median.array[1:dim(wl)[1],]), type="l")
plot(median.array[,18], type="l")


#Max 10 peaks per array
#Get top 10 maximum values and find if they are larger than the baseline
for (i in 1:dim(median.array)[2]) {
  temp.array<-as.vector(median.array[,i])
  #plot(temp.array, type = "l")
  for (j in 1:10) {
    
    max.index<-which(temp.array==max(temp.array))

    if(max.index!= 1|max.index!=length(temp.array)){
    index.1<-max(0, (max.index-30))
    index.2<-max(0, (max.index-2)) #Increase to avoid repetition
    index.3<-min(length(temp.array), (max.index+2))
    index.4<-min(length(temp.array), (max.index+30))
    
    if(temp.array[max.index] > 2*median(c(temp.array[index.1:index.2],temp.array[index.3:index.4] ))){
      anomalies<- c(anomalies,length(temp.array)*(i-1)+max.index)
      temp.array[index.2:index.3]<-
        median(c(temp.array[index.1:index.2],temp.array[index.3:index.4] ))
    }
    }else{
      temp.array[max.index]<- median(temp.array)  
    }
    
  }
}

plot(as.vector(median.array[1:dim(wl)[1],]), type="l")
points(anomalies, rep(100, length(anomalies)), col="red")

plots<-list()
anomaly.rank<-vector()
star<-"Tabby"
pcc.array<-matrix(NA, ncol = length(anomalies), nrow = dim(image.array)[1])

for (i in 1:length(anomalies)) {
  
  index <- which(wl==as.vector(wl)[anomalies[i]], arr.ind = TRUE)
  anomaly.rank[i] <- as.vector(median.array)[anomalies[i]] * as.vector(sd.array)[anomalies[i]]
  
  index.left<- index[1]-100
  index.right<- index[1]+100
  
  if(index.left<0) index.left<-1
  if(index.right>dim(image.array)[2]) index.right<-dim(image.array)[2]
  
  anomalies.left<-anomalies[i]-100
  anomalies.right<-anomalies[i]+100
  
  if(anomalies.left<0) anomalies.left<-1
  if(anomalies.right> length(as.vector(median.array))) anomalies.right<-length(as.vector(median.array))
  
  y<-as.vector(median.array)[anomalies.left:anomalies.right]
  #x<-c(1:length(y))
  x<-as.vector(wl)[anomalies.left:anomalies.right]
  x<-x/10

  array.plot <- image.array[ ,index.left:index.right, index[2] ]
  array.plot<-melt(array.plot)
  colnames(array.plot)<-c("time","wl","signal")
  
  array.plot$wl<-as.vector(wl)[anomalies.left+array.plot$wl-1]
  array.plot$wl<-array.plot$wl/10
  
  plot.time<- data.frame(image.array[,index[1], index[2]])
  pcc.array[,i]<-plot.time[,1]
  colnames(plot.time)<-"signal"
  plot.time$time <- date.julian
    

  
  #plots[[i]]<-
  plots[[i]]<-
    ggarrange(
      ggplot()+
        geom_line(data=array.plot, aes(wl,signal, group=time), alpha=0.2)+
        geom_line(aes(x ,y), alpha=1,colour="red", size=0.5)+
        geom_text_repel(aes(x[index[1]-index.left+1],y[index[1]-index.left+1], label= paste(round(as.vector(wl)[anomalies[i]]/10,2), "nm")  ), 
                        colour="red",
                        nudge_y = max(array.plot$signal)/2,
                        nudge_x = 20,
                        direction = "y",
                        arrow = arrow(length = unit(0.01, "npc"), type = "closed", ends = "first"))+
        labs(title = paste("Anomaly", i, "of",star  ))+
        
        theme_minimal(),
      
      ggplot(plot.time)+
        geom_line(aes(time, signal), col="red")+
        geom_point(aes(time, signal), col="red")+
        theme_minimal(),
      
      ncol=1, nrow = 2
      
    )

 
}

plots<-plots[order(anomaly.rank, decreasing = TRUE)]


#Anomalies correlation
pcc <- cor(pcc.array)
pcc<-melt(pcc)

pcc.plot<-ggplot(pcc)+
  geom_tile(aes(Var1, Var2, fill=value))+
  scale_fill_gradient2(low = "blue", high = "red", mid = "white", 
                       midpoint = 0, limit = c(-1,1), space = "Lab", 
                       name="Pearson's\nCorrelation") +
  xlim(1,length(anomalies))+
  ylim(1, length(anomalies))+
  theme_minimal()

##Plotting all anomalies in a pdf and top 6 in pdfs
pdf<-list()
pdf[[1]]<-ggarrange( plotlist = plots[c(1:min(length(anomalies), 6))])
try(pdf[[2]]<-ggarrange( plotlist = plots[c(7:min(length(anomalies), 12))]))
try(pdf[[3]]<-ggarrange( plotlist = plots[c(13:min(length(anomalies), 18))]))
try(pdf[[4]]<-ggarrange( plotlist = plots[c(19:min(length(anomalies), 24))]))

for (i in 1:length(pdf)) {
  ggsave(paste("/home/nacho/SETI/", star,"_",i,".pdf",sep = ), plot = pdf[[i]])
}
ggsave(paste("/home/nacho/SETI/", star,"_HM_",i,".pdf",sep = ), plot = pcc.plot)


##  Run in batch mode -----
results.folder<-"/home/nacho/SETI/Results/"

files<-read.csv("/home/nacho/SETI/apf_log.txt", sep = " ", header = FALSE)
files<-files[-1,]
files<-files[,1:8]
files$reduced<-gsub(".*\\/r", "TRUE", files$V1)
files$reduced<-gsub("TRUE.*", "TRUE", files$reduced)
files<-files[files$reduced=="TRUE", ]

wl<-readFITS("/home/nacho/SETI/apf_wav.fits")
wl<-wl$imDat

stars<- unique(as.character(files$V3))
stars.n<-vector()
date<-vector()

for (i in 1:length(stars)) {
  stars.n[i]<-nrow(files[files$V3==as.character(stars[i]),])
  
}
stars<-stars[order(stars.n, decreasing = TRUE)]
stars.n<-stars.n[order(stars.n, decreasing = TRUE)]

stars<-stars[which(stars.n>=5)]


##Check length
for (i in 140:length(stars)) {
  gc()
  #Add normalization step
  #for (i in 1:4) { #Test on 4 files
  temp.star.list <- files[files$V3==stars[i],]
  
  temp.star.list$link<-gsub("gs:\\/", "https:\\/\\/storage.googleapis.com", temp.star.list$V1)
  temp.star.list$filename<-gsub(".*\\/","",temp.star.list$V1)
  counter<-0
  image.array<-array(data = NA, dim = c(nrow(temp.star.list), 4608, 79))
  date<-vector()

  for (j in 1:nrow(temp.star.list)) {
    
    url = temp.star.list$link[j]
    url.size <- getURL(url, nobody=1L, header=1L)
    url.size <- gsub(".*length: ", "", url.size)
    url.size<-as.numeric(gsub("\r.*","", url.size))
    
    if(url.size<1500000){
    
      download.file(temp.star.list$link[j], "/home/nacho/SETI/temp.fits", quiet = TRUE)
      counter<-counter+1
      temp.fits<-readFITS("/home/nacho/SETI/temp.fits")
      date[counter]<-temp.fits$header[25]
      image.array[counter,,]<-temp.fits$imDat
    }
      }
  
  
    if(counter>=5){
      image.array<-image.array[1:counter,,]
      
      #Normalization
      photons_norm<-vector()
      for (timeevents in 1:dim(image.array)[1]) {
        photons_norm[timeevents]<-sum(image.array[timeevents,,])
        
      }
      photons_norm<-photons_norm/min(photons_norm)
      
      for (timeevents in 1:dim(image.array)[1]) {
        image.array[timeevents,,]<-image.array[timeevents,,,drop=FALSE]/photons_norm[timeevents]
        
      }
      
    

      #Spike finder
      mean.array<-matrix(data = NA, nrow = dim(image.array)[2], ncol = dim(image.array)[3])
      sd.array<-matrix(data = NA, nrow = dim(image.array)[2], ncol = dim(image.array)[3])
      median.array<-matrix(data = NA, nrow = dim(image.array)[2], ncol = dim(image.array)[3])
      
      for (k in 1:dim(mean.array)[2]) {
        mean.array[,k]<-apply(image.array[,,k], 2, mean)  
        sd.array[,k]<-apply(image.array[,,k], 2, sd)
        median.array[,k]<-apply(image.array[,,k], 2, median)  
      }
      
      date<-gsub(".*=..", "",date)
      date<-gsub(". / UT.*", "",date)
      date<-as.POSIXct(date,format="%Y-%m-%dT%H:%M:%S")
      date.julian<-julian.POSIXt(date)
      date.julian<-(date.julian-min(date.julian))
      date.julian<-as.numeric(date.julian)
      
      #Scan segments of wl for the intensity over time 
      segment.size<-100
      spec.size <- max(wl)-min(wl)
      image.array.fit<-image.array[,1:nrow(wl),]
      image.array.fit[image.array.fit<0]<-0
      try(rm(output.segment))
      #For loop over time
      for (l in 1:dim(image.array)[1]) {
        dummy.array<-image.array.fit[l,,]
        segment.array<-data.frame(matrix(NA, ncol = 4, nrow = round(spec.size/segment.size)))
        colnames(segment.array)<-c("wl_start", "wl_end", "median","time")
        segment.array$wl_start[1]<-min(wl)
        segment.array$wl_end[1]<-segment.array$wl_start[1]+segment.size
        segment.array$median[1]<- median(dummy.array[ which(wl>=segment.array$wl_start[1] & wl<segment.array$wl_end[1] )] )
        segment.array$time<-date.julian[l]
        
        for (m in 2:nrow(segment.array)) {
          segment.array$wl_start[m]<-segment.array$wl_end[m-1]  
          segment.array$wl_end[m]<-segment.array$wl_start[m]+segment.size
          segment.array$median[m]<- median(dummy.array[ which(wl>=segment.array$wl_start[m] & wl<segment.array$wl_end[m])] )
        }
        if(!exists("output.segment")){
          output.segment<-segment.array
        }else{
          output.segment<-rbind(output.segment, segment.array)
        }
        
      }
      
      segments<-ggplot(output.segment)+
        geom_line(aes(wl_start, median))+
        facet_wrap(~round(time,3))+
        theme_minimal()+
        theme(axis.text.x = element_text(angle = 60, hjust = 1))
        
      
      ggsave(paste(results.folder, stars[i],"_Segments",".pdf",sep ="" ), plot = segments,
             width = 21, height = 29.7, units = "cm")
      
      #Second section  ###To check all items of the loops
      median.array<-median.array[-nrow(median.array),]
      anomalies<-vector()

      #Max 10 peaks per array
      #Get top 10 maximum values and find if they are larger than the baseline
      for (i2 in 1:dim(median.array)[2]) {
        temp.array<-as.vector(median.array[,i2])
        #plot(temp.array, type = "l")
        for (j2 in 1:10) {
          
          max.index<-which(temp.array==max(temp.array))
          
          if(max.index!= 1|max.index!=length(temp.array)){
            index.1<-max(0, (max.index-30))
            index.2<-max(0, (max.index-2)) 
            index.3<-min(length(temp.array), (max.index+2))
            index.4<-min(length(temp.array), (max.index+30))
            
            if(temp.array[max.index] > 2*median(c(temp.array[index.1:index.2],temp.array[index.3:index.4] ))){
              anomalies<- c(anomalies,length(temp.array)*(i2-1)+max.index)
              temp.array[max(0, (max.index-7)):min(length(temp.array), (max.index+7))]<-
                median(c(temp.array[index.1:index.2],temp.array[index.3:index.4] ))
            }
          }else{
            temp.array[max.index]<- median(temp.array)  
          }
          
        }
      }
      
      #plot(as.vector(median.array[1:dim(wl)[1],]), type="l")
      #points(anomalies, rep(1000, length(anomalies)), col="red")
      if(length(anomalies)>0){
      plots<-list()
      anomaly.rank<-vector()

      pcc.array<-matrix(NA, ncol = length(anomalies), nrow = dim(image.array)[1])
      try(rm(df.save.bind))
      for (i2 in 1:length(anomalies)) {
        
        index <- which(wl==as.vector(wl)[anomalies[i2]], arr.ind = TRUE)
        if(nrow(index)==1){
        
        anomaly.rank[i2] <- as.vector(median.array)[anomalies[i2]] * as.vector(sd.array)[anomalies[i2]]
        
        index.left<- index[1]-100
        index.right<- index[1]+100
        
        if(index.left<=0) index.left<-1
        if(index.right>dim(image.array)[2]) index.right<-dim(image.array)[2]
        
        anomalies.left<-anomalies[i2]-100
        anomalies.right<-anomalies[i2]+100
        
        if(anomalies.left<=0) anomalies.left<-1
        if(anomalies.right> length(as.vector(median.array))) anomalies.right<-length(as.vector(median.array))
        
        y<-as.vector(median.array)[anomalies.left:anomalies.right]
        #x<-c(1:length(y))
        x<-as.vector(wl)[anomalies.left:anomalies.right]
        x<-x/10
        
        #Check wl jump
        
        array.plot <- image.array[ ,index.left:index.right, index[2] ]
        array.plot<-melt(array.plot)
        colnames(array.plot)<-c("time","wl","signal")
        
        array.plot$wl<-as.vector(wl)[anomalies.left+array.plot$wl-1]
        array.plot$wl<-array.plot$wl/10
        
        df.save<-array.plot
        df.save$star<-stars[i]
        df.save$anomaly.wl<-as.vector(wl)[anomalies[i2]]
        df.save$anomaly.score<-anomaly.rank[i2]
        df.save$julian.time<-NA
        df.save$julian.time.norm<-NA
        df.save$anomaly.id<-paste(stars[i],"_A" ,anomalies[i2], sep = "")
        
        
        for (time in 1:dim(image.array)[1]) {
          df.save$julian.time.norm[df.save$time==time]<-date.julian[time]
          df.save$julian.time[df.save$time==time]<-as.numeric(julian.POSIXt(date[time]))
          df.save$time[df.save$time==time]<-date[time]
        }
        if(!exists("df.save.bind")){
          df.save.bind<-df.save
        }else{
          df.save.bind<-rbind(df.save.bind, df.save)
        }
        
        plot.time<- data.frame(image.array[,index[1], index[2]])
        pcc.array[,i2]<-plot.time[,1]
        colnames(plot.time)<-"signal"
        plot.time$time <- date.julian
        
        plots[[i2]]<-
          ggarrange(
            ggplot()+
              geom_line(data=array.plot, aes(wl,signal, group=time), alpha=0.2)+
              geom_line(aes(x ,y), alpha=1,colour="red", size=0.5)+
              geom_text_repel(aes(x[index[1]-index.left+1],y[index[1]-index.left+1], label= paste(round(as.vector(wl)[anomalies[i2]]/10,2), "nm")  ), 
                              colour="red",
                              nudge_y = max(array.plot$signal)/2,
                              nudge_x = 20,
                              direction = "y",
                              arrow = arrow(length = unit(0.01, "npc"), type = "closed", ends = "first"))+
              labs(title = paste("Anomaly", i2, "of",stars[i]  ))+
              
              theme_minimal(),
            
            ggplot(plot.time)+
              geom_line(aes(time, signal), col="red")+
              geom_point(aes(time, signal), col="red")+
              theme_minimal(),
            
            ncol=1, nrow = 2
            
          )}
      }
      
      plots<-plots[order(anomaly.rank, decreasing = TRUE)]
      
      #Anomalies correlation
      pcc <- cor(pcc.array)
      pcc<-melt(pcc)
      
      # for(index.pcc in 1:length(anomalies)){
      #   df.save.bind$anomaly.score[anomalies[index.pcc]]<-df.save.bind$anomaly.score[anomalies[index.pcc]]/abs(pcc$value[pcc$Var1==index.pcc])
      # }
      
      pcc.plot<-ggplot(pcc)+
        geom_tile(aes(Var1, Var2, fill=value))+
        scale_fill_gradient2(low = "blue", high = "red", mid = "white", 
                             midpoint = 0, limit = c(-1,1), space = "Lab", 
                             name="Pearson's\nCorrelation") +
        xlim(1,length(anomalies))+
        ylim(1, length(anomalies))+
        theme_minimal()
      
      ##Plotting all anomalies in a pdf and top 6 in pdfs
      if(length(anomalies)>0){
      pdf<-list()
      pdf[[1]]<-ggarrange( plotlist = plots[c(1:min(length(anomalies), 6))])
      if(max(length(anomalies), 6)==length(anomalies))  pdf[[2]]<-ggarrange( plotlist = plots[c(7:min(length(anomalies), 12))])
      if(max(length(anomalies), 12)==length(anomalies))  pdf[[3]]<-ggarrange( plotlist = plots[c(13:min(length(anomalies), 18))])
      if(max(length(anomalies), 18)==length(anomalies))  pdf[[4]]<-ggarrange( plotlist = plots[c(19:min(length(anomalies), 24))])
      if(max(length(anomalies), 24)==length(anomalies))  pdf[[5]]<-ggarrange( plotlist = plots[c(25:min(length(anomalies), 30))])

      for (i2 in 1:length(pdf)) {
        ggsave(paste(results.folder, stars[i],"_Anomalies",i2,".pdf",sep ="" ), plot = pdf[[i2]],
               width = 21, height = 29.7, units = "cm")

      }
      if(length(anomalies>=2)){
       ggsave(paste(results.folder, stars[i],"_PCC",".pdf",sep ="" ),  plot = pcc.plot,
              width = 21, height = 29.7, units = "cm")
  }     
       write.csv(df.save.bind, paste(results.folder, stars[i],"_Anomalies_DF",".csv",sep ="" ))
      }
      }
    }#Counter 5
  print(paste("Done with star", stars[i]))
}




## Meta analysis & Noise computation for star

result.files<-list.files("/home/nacho/SETI/Results/")
csvs<-grep("csv",result.files)

for (i in 1:length(csvs)) {
  dummy<-read.csv(paste("/home/nacho/SETI/Results/", result.files[csvs[i]], sep = "" ))
  if(!exists("df.meta")){
    df.meta<-dummy
  }else{
    df.meta<-rbind(df.meta,dummy)
  }
}
write.csv(df.meta,"/home/nacho/SETI/Results/df.meta.csv")

df.meta<-read.csv("/home/nacho/SETI/Results/df.meta.csv", stringsAsFactors = FALSE)
star.id<-unique(df.meta$star)
anomaly.id<-unique(df.meta$anomaly.id)

pb<-txtProgressBar(min = 1, max = length(anomaly.id), initial = 1)
anomaly.wl<-vector()
for (i in 1:length(anomaly.id)) {
  setTxtProgressBar(pb,i)
  dummy<-df.meta$wl[df.meta$anomaly.id==anomaly.id[i]]
  dummy<-unique(dummy)
  anomaly.wl[i]<-mean(dummy)
}

hist.wl<-hist(anomaly.wl, breaks = 1000)
counts<-hist.wl$counts
breaks<-hist.wl$breaks

counters.to.get<-which(counts>0 & counts<=2)

special.anomalies<-vector()
  
for (i in 1:length(counters.to.get)) {
 special.anomalies<-c(special.anomalies,anomaly.id[which(anomaly.wl>breaks[counters.to.get[i]] & anomaly.wl<breaks[counters.to.get[i]+1] )])
}
