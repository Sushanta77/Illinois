
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
