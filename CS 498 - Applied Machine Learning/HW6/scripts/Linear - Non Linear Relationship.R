#Below is the Linear & Non-Linear Relationship of the Auto Dataset (Final)
par(mfrow=c(1,1))
auto <- read.csv('auto.csv')
auto$horsepower_int <- as.integer(as.character(auto$horsepower))
auto$horsepower_int2 <- auto$horsepower_int^2
auto$horsepower_int3 <- auto$horsepower_int^3
auto$horsepower_int4 <- auto$horsepower_int^4
auto$horsepower_int5 <- auto$horsepower_int^5
auto$horsepower_int6 <- auto$horsepower_int^6
auto$horsepower_int7 <- auto$horsepower_int^7
auto$horsepower_int8 <- auto$horsepower_int^8
auto$horsepower_int9 <- auto$horsepower_int^9
auto$horsepower_int10 <- auto$horsepower_int^10
auto_sort <- auto[order(auto$horsepower_int),c(1,10,11,12,13,14,15,16,17,18,19)]
fit_lm <- lm(mpg~horsepower_int,data=auto_sort)
fit_lm2 <- lm(mpg~horsepower_int+horsepower_int2,data=auto_sort)
fit_lm5 <- lm(mpg~horsepower_int+horsepower_int2+horsepower_int3+horsepower_int4+horsepower_int5,data=auto_sort)
fit_lm6 <- lm(mpg~horsepower_int+horsepower_int2+horsepower_int3+horsepower_int4+horsepower_int5+horsepower_int6,data=auto_sort)
fit_lm7 <- lm(mpg~horsepower_int+horsepower_int2+horsepower_int3+horsepower_int4+horsepower_int5+horsepower_int6+horsepower_int7,data=auto_sort)
fit_lm8 <- lm(mpg~horsepower_int+horsepower_int2+horsepower_int3+horsepower_int4+horsepower_int5+horsepower_int6+horsepower_int7+horsepower_int8,data=auto_sort)
fit_lm9 <- lm(mpg~horsepower_int+horsepower_int2+horsepower_int3+horsepower_int4+horsepower_int5+horsepower_int6+horsepower_int7+horsepower_int8+horsepower_int9,data=auto_sort)
fit_lm10 <- lm(mpg~horsepower_int+horsepower_int2+horsepower_int3+horsepower_int4+horsepower_int5+horsepower_int6+horsepower_int7+horsepower_int8+horsepower_int9+horsepower_int10,data=auto_sort)
pred <- 39.9359 + (-0.1578)*auto_sort$horsepower_int
pred2 <- 56.900100 + (-0.466190)*auto_sort$horsepower_int+ (0.001231 )*auto_sort$horsepower_int2
pred5 <- -3.223e+01  + (3.700e+00)*auto_sort$horsepower_int+(-7.142e-02)*auto_sort$horsepower_int2+(5.931e-04)*auto_sort$horsepower_int3+(-2.281e-06)*auto_sort$horsepower_int4+(3.330e-09)*auto_sort$horsepower_int5
pred6 <- -1.621e+02  + (1.124e+01)*auto_sort$horsepower_int+(-2.436e-01)*auto_sort$horsepower_int2+(2.580e-03)*auto_sort$horsepower_int3+(-1.453e-05)*auto_sort$horsepower_int4+(4.173e-08)*auto_sort$horsepower_int5+(-4.803e-11)*auto_sort$horsepower_int6
pred7 <- -4.891e+02  + (3.325e+01)*auto_sort$horsepower_int+(-8.476e-01)*auto_sort$horsepower_int2+(1.135e-02)*auto_sort$horsepower_int3+(-8.755e-05)*auto_sort$horsepower_int4+(3.914e-07)*auto_sort$horsepower_int5+(-9.429e-10)*auto_sort$horsepower_int6+(9.472e-13)*auto_sort$horsepower_int7
plot(auto_sort$horsepower_int,auto_sort$mpg,col='deepskyblue4',xlab='HorsePower',ylab='Mileage Per Gas')
lines(auto_sort$horsepower_int,pred,col='blue',lwd=3)
lines(auto_sort$horsepower_int,pred2,col='green',lwd=3)
lines(auto_sort$horsepower_int,pred5,col='magenta',lwd=3)
lines(auto_sort$horsepower_int,pred6,col='red',lwd=3)
lines(auto_sort$horsepower_int,pred7,col='orange',lwd=3)



fit_lm2 <- lm(mpg~horsepower_int+horsepower_int2,data=auto_sort)


pred2 <- 44.1041033 + (-0.2493509)*auto_sort$horsepower_int+(0.0004195)*auto_sort$horsepower_int2
lines(auto_sort$horsepower_int,pred2,col='blue',lwd=3)

