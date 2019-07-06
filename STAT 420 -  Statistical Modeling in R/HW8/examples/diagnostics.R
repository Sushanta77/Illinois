
sim_1 = function (sample_size=500){
  x = runif(n = sample_size) * 5
  epsilon = rnorm(n = sample_size,mean = 0,sd = 1)
  y = 3 + 5 * x + epsilon
  data.frame(x,y)
}

sim_2 = function(sample_size = 500){
  x = runif(n=sample_size) * 5
  epsilon = rnorm(sample_size,mean = 0, sd = x)
  y = 3 + 5 * x + epsilon
  data.frame(x,y)
}

sim_3 = function(sample_size = 500){
  x = runif(n=sample_size) * 5
  epsilon = rnorm(sample_size,mean = 0, sd = 5)
  y = 3 + 5 * x^2 + epsilon
  data.frame(x,y)
}


#Generated good data
set.seed(42)
sim_data_1 = sim_1()



#Scatterplot with fitted slr
plot(y~x,
     data=sim_data_1,
     col = "grey",
     main = "Data from Model 1")
model_1 = lm(y~x,data=sim_data_1)
abline(model_1,col = "darkorange",lwd=3)
#fitted versus residuals
plot(fitted(model_1),resid(model_1),col = "grey",main = "Fitted Vs Residuals")
abline(h = 0, col = "darkorange",lwd = 3)

set.seed(42)
sim_data_2 = sim_2()
#Scatterplot with different sigma
plot(y~x,
     data=sim_data_2,
     col = "grey",
     main = "Data from Model 2")
model_2 = lm(y~x,data=sim_data_2)
abline(model_2,col = "darkorange",lwd=3)
#fitted versus residuals
plot(fitted(model_2),resid(model_2),col = "grey",main = "Fitted Vs Residuals")
abline(h = 0, col = "darkorange",lwd = 3)


set.seed(42)
sim_data_3 = sim_3()
#Scatterplot with fitted qudratic function
plot(y~x,
     data=sim_data_3,
     col = "grey",
     main = "Data from Model 3")
model_3 = lm(y~x,data=sim_data_3)
abline(model_3,col = "darkorange",lwd=3)
#fitted versus residuals
plot(fitted(model_3),resid(model_3),col = "grey",main = "Fitted Vs Residuals")
abline(h = 0, col = "darkorange",lwd = 3)


library(lmtest)
bptest(model_1)
bptest(model_2)
bptest(model_3)

par(mfrow=c(1,3))
hist(resid(model_1),
     col = "darkorange",
     border = "white",
     breaks = 20)
hist(resid(model_2),
     col = "darkorange",
     border = "white",
     breaks = 20)
hist(resid(model_3),
     col = "darkorange",
     border = "white",
     breaks = 20)


#Plot the Q-Q Plot
x = rnorm(100,mean = 0, sd = 1)
par(mfrow=c(1,2))
qqnorm(x,col="grey",xlab = "Therotical Quantiles",ylab = "Sample Quantiles")
qqline(x,col="dodgerblue",lwd=2,lty=2)


#Let's see the Q-Q plot in case of samples drawn from the Normal Distribution
par(mfrow=c(1,3))
set.seed(42)
qqnorm(rnorm(10),col="grey",xlab = "Therotical Quantiles",ylab = "Sample Quantiles")
qqline(rnorm(10),col="dodgerblue",lwd=2,lty=2)
qqnorm(rnorm(25),col="grey",xlab = "Therotical Quantiles",ylab = "Sample Quantiles")
qqline(rnorm(25),col="dodgerblue",lwd=2,lty=2)
qqnorm(rnorm(100),col="grey",xlab = "Therotical Quantiles",ylab = "Sample Quantiles")
qqline(rnorm(100),col="dodgerblue",lwd=2,lty=2)



#Let's see the Q-Q plot in case of samples drawn from the T Distribution
par(mfrow=c(1,3))
set.seed(42)
qqnorm(rt(10,df=4),col="grey",xlab = "Therotical Quantiles",ylab = "Sample Quantiles")
qqline(rt(10,df=4),col="dodgerblue",lwd=2,lty=2)
qqnorm(rt(25,df=4),col="grey",xlab = "Therotical Quantiles",ylab = "Sample Quantiles")
qqline(rt(25,df=4),col="dodgerblue",lwd=2,lty=2)
qqnorm(rt(100,df=4),col="grey",xlab = "Therotical Quantiles",ylab = "Sample Quantiles")
qqline(rt(100,df=4),col="dodgerblue",lwd=2,lty=2)


#Data without assumption violation
par(mfrow=c(1,1))
qqnorm(resid(model_1),col="darkgrey",xlab = "Normal Q-Q Plot, model_1")
qqline(resid(model_1),col="dodgerblue",lty=2,lwd=2)

#Data with non constant variance
par(mfrow=c(1,1))
qqnorm(resid(model_2),col="darkgrey",xlab = "Normal Q-Q Plot,model_2")
qqline(resid(model_2),col="dodgerblue",lty=2,lwd=2)


#Data with quadratic form
par(mfrow=c(1,1))
qqnorm(resid(model_3),col="darkgrey",xlab = "Normal Q-Q Plot,model_3")
qqline(resid(model_3),col="dodgerblue",lty=2,lwd=2)

#Saphiro wiki Test
set.seed(42)
shapiro.test(rnorm(25))
shapiro.test(rexp(25))

#Shapiro Test for our Model
shapiro.test(resid(model_1))
shapiro.test(resid(model_2))
shapiro.test(resid(model_3))

#Additive Model
model_add_hp = lm(mpg~hp+am, data=mtcars)
plot(fitted(model_add_hp),resid(model_add_hp),col="darkgrey",xlab="Fitted",ylab="Residuals",pch=20,cex=2)
abline(h=0,col="darkorange",lwd=2)

#Runs the BP Test which is analog to the Reidual Vs Fitted Line
bptest(model_add_hp)

qqnorm(resid(model_add_hp),col = "darkgrey",lwd=2)
qqline(resid(model_add_hp),col = "dodgerblue",lwd=2,lty=2)
shapiro.test(resid(model_add_hp))


#-----------------------------------------------------------------------------------------------------------------------------
#
# Larger Mpg Data set
#
#-----------------------------------------------------------------------------------------------------------------------------
data = read.table("https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data",
                  quote = "\"",
                  comment.char = "",
                  stringsAsFactors = FALSE)
#Change the column name of the data set autompg
colnames(data) = c("mpg","cyl","disp","hp","wt","acc","year","origin","name")
#remove missing data stored as "?"
data = subset(data,data$hp != "?")
#remove the plymouth, as it causes issues
data = subset(data,data$name != "plymouth reliant")
#Assign the rowname, based on the engine, year and name
rownames(data) = paste(data$cyl,"cylinder",data$year,data$name)
#remove the variable for the name
data = subset(data,select = c("mpg","cyl","disp","hp","wt","acc","year","origin"))
#Change the horsepower from character to name
data$hp = as.numeric(data$hp)
#Creata a dummy variable for foreign vs domestic cars, domestic = 1
data$domestic = as.numeric(data$origin == 1)
#remove the 3 and 5 cylinder cars, as they arr very rare
data = data[data$cyl != 5,]
data = data[data$cyl != 3,]
#the following line would verify the remaining cylinder possibilities are 4,6,8
#unique(data$cyl)
#change cyl to a factor variable
data$cyl = as.factor(data$cyl) 

big_model = lm(mpg~disp * hp * domestic,data=data)

#Check the Residual Vs Fitted Model, to see what's going on
plot(fitted(big_model),resid(big_model),col="darkgrey",lwd=2,lty=20)
abline(h = 0, col = "dodgerblue",lwd = 2, lty = 2)
bptest(big_model)

#Check the Q-Q Plot 
qqnorm(resid(big_model),col = "darkgrey", lwd = 2, lty = 20)
qqline(resid(big_model),col = "dodgerblue", lwd = 2, lty = 2)

shapiro.test(resid(big_model))
