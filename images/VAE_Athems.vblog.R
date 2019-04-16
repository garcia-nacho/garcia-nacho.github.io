#VAE
#nacho garcia 2019
#garcia.nacho@gmail.com

#Library loading
library(keras)
library(tensorflow)
library(tfdatasets)
library(tuneR)
library(ggplot2)

path<-"/home/nacho/Downloads/Midi/NationalAnthems/"
files<-list.files(path)
load("/home/nacho/VAE Athems/Anthems.RData")
TS<-read.csv("/home/nacho/VAE Athems/TS.csv")

CO<-3000

channels<-9


#Cutoff time
t2<-0
for (i in 1:length(df)){
  t2 <- c(t2,max(df[[i]]$time))
}
t2<-t2[-1]
total.time<-t2
hist(total.time, breaks = 60)
#CO<-50000

t3<-0
for (i in 1:length(df)){
  t3 <- c(t3,min(df[[i]]$length))
}
t3<-t3[-1]
hist(t3, breaks = 60)

#Check maximum element per track
t4<-0
for (i in 1:length(df)){
  tracks<-unique(df[[i]]$track)
  for (j in 1:length(tracks)) {
  t4 <- c(t4, length(df[[i]]$time[df[[i]]$track==tracks[j]]))  
  }
  
}
t4<-t4[-1]
hist(t4, breaks = 60, main="Number of Notes", xlab = "Notes per Track")
Notes.N<-150

#Min and max notes
n.min<-0
for (i in 1:length(df)){
  n.min <- c(n.min,min(df[[i]]$note))
}
n.min<-n.min[-1]
hist(n.min, breaks = 60)
Note.min<-24

n.max<-0
for (i in 1:length(df)){
  n.max <- c(n.max,max(df[[i]]$note))
}
n.max<-n.max[-1]
hist(n.max, breaks = 60)
Note.max<-101

ggplot(df[[1]])+
  geom_line(aes(x=time,y=note, colour=as.character(track)))+
  theme_minimal()

#Array creation
df.array <- array(data=0, dim=c(length(df),Notes.N,3, channels))



#Array filling
for (i in 1:length(df)) {
  tracks<-unique(df[[i]]$track)
  for (j in 1:min(length(tracks),channels)) {
    #Note
    
    dummy.note<-df[[i]]$note[df[[i]]$track==tracks[j]]
    dummy.time<-df[[i]]$time[df[[i]]$track==tracks[j]]+1
    dummy.length<-df[[i]]$length[df[[i]]$track==tracks[j]]
    
    
    # df.array[i,1:min(length(dummy.note),Notes.N),1,j]<-dummy.note[1:min(length(dummy.note),Notes.N)]
    # df.array[i,1:min(length(dummy.length),Notes.N),2,j]<-dummy.length[1:min(length(dummy.length),Notes.N)]
    # df.array[i,1:min(length(dummy.time),Notes.N),3,j]<-dummy.time[1:min(length(dummy.time),Notes.N)]


    df.array[i,1:min(length(dummy.note),Notes.N),1,j]<-dummy.note[1:min(length(dummy.note),Notes.N)]
    df.array[i,1:min(length(dummy.length),Notes.N),2,j]<-dummy.length[1:min(length(dummy.length),Notes.N)]
    df.array[i,1:min(length(dummy.time),Notes.N),3,j]<-dummy.time[1:min(length(dummy.time),Notes.N)]
  }

}

#Keras model VAE
filters <- 36
intermediate_dim <- 256
batch_size <- 10
epoch <- 100
latent_dim <- 2
epsilon_std <- 1.0
Reg_factor <- 0.5
Loss_factor<-0.5

#### Model
dimensions<-dim(df.array)
dimensions<-dimensions[-1]

Input <- layer_input(shape = dimensions)

Notes<- Input %>%
  layer_conv_2d(filters=filters, kernel_size=c(5,1), activation='relu', padding='same',strides=1,data_format='channels_last')%>% 
  layer_conv_2d(filters=filters*2, kernel_size=c(5,1), activation='relu', padding='same',strides=1,data_format='channels_last')%>%
  layer_conv_2d(filters=filters*4, kernel_size=c(5,3), activation='relu', padding='same',strides=1,data_format='channels_last')%>%
  layer_conv_2d(filters=filters*8, kernel_size=c(5,3), activation='relu', padding='same',strides=1,data_format='channels_last')%>%
  layer_flatten()

hidden <- Notes %>% layer_dense( units = intermediate_dim, activation = "sigmoid") %>% 
          layer_dense( units = round(intermediate_dim/2), activation = "sigmoid") %>% 
          layer_dense( units = round(intermediate_dim/4), activation = "sigmoid")

z_mean <- hidden %>% layer_dense( units = latent_dim)
z_log_var <- hidden %>% layer_dense( units = latent_dim)

sampling <- function(args) {
  z_mean <- args[, 1:(latent_dim)]
  z_log_var <- args[, (latent_dim + 1):(2 * latent_dim)]
  
  epsilon <- k_random_normal(
    shape = c(k_shape(z_mean)[[1]]),
    mean = 0.,
    stddev = epsilon_std
  )
  z_mean + k_exp(z_log_var) * epsilon
}

z <- layer_concatenate(list(z_mean, z_log_var)) %>% layer_lambda(sampling)

output_shape <- c(10, dimensions)  
Output<- z %>%
  layer_dense( units = round(intermediate_dim/4), activation = "sigmoid") %>% 
  layer_dense( units = round(intermediate_dim/2), activation = "sigmoid") %>%
  layer_dense(units = intermediate_dim, activation = "sigmoid") %>%
  layer_dense(units = prod(150,3,filters*8), activation = "relu") %>%
  layer_reshape(target_shape = c(150,3,filters*8)) %>%
  layer_conv_2d_transpose(filters=filters*8, kernel_size=c(5,3), activation='relu', padding='same',strides=1,data_format='channels_last')%>%
  layer_conv_2d_transpose(filters=filters*4, kernel_size=c(5,3), activation='relu', padding='same',strides=1,data_format='channels_last')%>%
  layer_conv_2d_transpose(filters=filters*2, kernel_size=c(5,1), activation='relu', padding='same',strides=1,data_format='channels_last')%>%
  layer_conv_2d_transpose(filters=9, kernel_size=c(5,1), activation='relu', padding='same',strides=1,data_format='channels_last')


# 
# Output<- z %>%
#   layer_dense(units = intermediate_dim, activation = "relu") %>%
#   layer_dense(units = prod(150,3,filters*8), activation = "relu") %>%
#   layer_reshape(target_shape = c(150,3,filters*8)) %>%
#   layer_conv_2d_transpose(filters=filters*8, kernel_size=c(5,3), activation='relu', padding='same',strides=1,data_format='channels_last')%>%
#   layer_conv_2d_transpose(filters=filters*4, kernel_size=c(5,3), activation='relu', padding='same',strides=1,data_format='channels_last')%>%
#   layer_conv_2d_transpose(filters=filters*2, kernel_size=c(5,1), activation='relu', padding='same',strides=1,data_format='channels_last')%>%
#   layer_conv_2d_transpose(filters=filters, kernel_size=c(5,1), activation='relu', padding='same',strides=2,data_format='channels_last')%>%
#   layer_conv_2d(filters=9, kernel_size=c(5,1), activation='softmax', padding='same',strides=2,data_format='channels_last')



# custom loss function
vae_loss <- function(x, x_decoded_mean_squash) {
  
  x <- k_flatten(x)
  
  x_decoded_mean_squash <- k_flatten(x_decoded_mean_squash)
  
  # xent_loss <- 9 * dim(df.array)[2] * dim(df.array)[3] *
  #   loss_binary_crossentropy(x, x_decoded_mean_squash)
  # 
  xent_loss <- Loss_factor*loss_mean_squared_logarithmic_error(x, x_decoded_mean_squash)
  
  kl_loss <- -Reg_factor * k_mean(1 + z_log_var - k_square(z_mean) - k_exp(z_log_var), axis = -1L)
  
  k_mean(xent_loss + kl_loss)
}


## variational autoencoder
vae <- keras_model(Input, Output)
vae %>% compile(optimizer = "rmsprop", loss = vae_loss)
summary(vae)

#Data generation
#Training/validation 10 test

val<- sample(dim(df.array)[1], 10)
xval<- df.array[val,,,]
xtrain<-df.array[-val,,,]

reset_states(vae)

date<-as.character(date())
logs<-gsub(" ","_",date)
logs<-gsub(":",".",logs)
logs<-paste("logs/",logs,sep = "")

history<-vae %>% fit(x= xtrain,
                     y=xtrain,
                     batch_size=batch_size,
                     epoch=epoch,
                     validation_data = list(xval, xval),
                     callbacks = callback_tensorboard(logs),
                     view_metrics=FALSE,
                     shuffle=TRUE)

tensorboard(logs)



  #Troubleshooting
  
  layer_name <- 'lambda_1'
  intermediate_layer_model <- keras_model(inputs = vae$input,
                                          outputs = get_layer(vae, layer_name)$output)
  intermediate_output <- predict(intermediate_layer_model, xtrain)
  intermediate_output
  
  
  #Plot Anthems 2D
  
  Anth2D <- predict(intermediate_layer_model, df.array, batch_size = 10)
  Anth2D <- as.data.frame(Anth2D)
  Anth2D$Country<-files[-73]
  Anth2D$Country<-gsub(".mid","",Anth2D$Country)
  
  ggplot(Anth2D)+
    geom_jitter(aes(V1,V2))+
    #stat_density_2d(geom = "raster", aes(V1,V2,fill = stat(density)), contour = FALSE)+
    #scale_fill_distiller(palette = 'RdYlBu')+
    geom_text(aes(V1,V2,label=Country),hjust=-0.1, vjust=0, size=2.5)+
    theme_minimal()
  
  #K-means
  wss<-0
  for (i in 2:20){
    wss[i] <- sum(kmeans(Anth2D[,1:2], centers = i)$withinss)
  
}
  plot(1:20, wss, type = "b", xlab = "Number of groups",   
       ylab = "Within groups sum of squares")
  km<-kmeans(Anth2D[,1:2], centers = 7)
  Anth2D$Cluster <- km$cluster 
  
  ggplot(Anth2D)+
    geom_jitter(aes(V1,V2, colour=as.character(Cluster)))+
    #stat_density_2d(geom = "raster", aes(V1,V2,fill = stat(density)), contour = FALSE)+
    #scale_fill_distiller(palette = 'RdYlBu')+
    geom_text(aes(V1,V2,label=Country),hjust=-0.1, vjust=0, size=2.5)+
    theme_minimal()
  
  #Hclust
  d_Anth2D <- dist(Anth2D[,1:2], method="euclidean")
  Anth2D_Dend<-hclust(d_Anth2D, method="complete")
  Anth2D_Dend$labels<-Anth2D$Country
  plot(Anth2D_Dend, cex=0.7)
  
  #Regeneration of sample #1
  Samp1.gen <- predict(vae, array(df.array[1,,,], dim = c(1,150,3,9)), batch_size = 10)
  
      Samp1.gen[1,,1,] <- round(Samp1.gen[1,,1,])
      Samp1.gen[1,,2,] <- round(Samp1.gen[1,,2,])
      Samp1.gen[1,,3,] <- round(Samp1.gen[1,,3,])
      
      to.plot <- as.data.frame(Samp1.gen[1,,,1])
      colnames(to.plot) <- c("note", "Length", "time")
      to.plot$Channel<-1
      
      for (i in 2:9) {
        dummy <- as.data.frame(Samp1.gen[1,,,i])
        colnames(dummy) <- c("note", "Length", "time")
        dummy$Channel<-i
        to.plot<-rbind(to.plot,dummy)
      }
      
  ggplot(to.plot)+
    geom_line(aes(x=time,y=note, colour=as.character(Channel)))+
    theme_minimal()

  
  # vae %>% fit(
  #   x_train, x_train, 
  #   shuffle = TRUE, 
  #   epochs = epochs, 
  #   batch_size = batch_size, 
  #   validation_data = list(x_test, x_test)
  # )  

vae(xtrain)
  