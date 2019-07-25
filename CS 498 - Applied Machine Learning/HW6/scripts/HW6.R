#-------------------------------------------Load the data set
column_name <- c("crim","zn","indus","chas","nox","rm","age","dis","rad","tax","ptratio","b","lstat","medv")
housing <- read.table("HW6/data/housing.data",col.names=column_name)
#-------------------------------------------Fit the Linear Data Model
fit <- lm (medv~crim+zn+indus+chas+nox+rm+age+dis+rad+tax+ptratio+b+lstat,data=housing)
plot(fit)
summary(fit)
#########################################################################################################
# 
# Residuals, Cooks Distance, Leverage ########  Before Outlier
#
#########################################################################################################
#-------------------------------------------Diagnostic Plot (Plot 1,3) #### Residuals
par(mfrow=c(2,2))
fit <- lm (medv~crim+zn+indus+chas+nox+rm+age+dis+rad+tax+ptratio+b+lstat,data=housing)
res <- fit$residuals
plot(fit,which=1)
text(predict(fit),res,ifelse( ((rownames(housing)==369 | rownames(housing)==372 | rownames(housing)==373) 
                                           | (!(res < -12 | res > 14)) ),"",rownames(housing) ),
     cex= 0.8,pos=3,col='red')
hist(res)

rstandards = rstandard(fit)
plot(predict(fit),rstandards,xlab="Fitted Values",ylab="Standardized Residuals",main="Standardize Residuals Vs Fitted")
text(predict(fit),rstandard(fit),ifelse((!(rstandards < -3 | rstandards > 3)),"",rownames(housing) ),
     cex= 0.8,pos=2,col='red')
hist(rstandards)

#-------------------------------------------Diagnostic Plot (Plot 4,5) #### Cooks Distaince, Leverage
par(mfrow=c(2,2))
plot(fit,which=4)
text(rownames(housing),cooks.distance(fit),ifelse( ((rownames(housing)==369 | rownames(housing)==373 | rownames(housing)==365) 
                                                                 | (cooks.distance(fit) < 0.04) ),"",rownames(housing) ),
     cex= 0.8,pos=3,col='red')
hist(cooks.distance(fit))
plot(fit,which=5)
text(hatvalues(fit),(rstandard(fit)),ifelse( ((rownames(housing)==369 | rownames(housing)==373 | rownames(housing)==365) 
                                                          | (!(rstandard(fit) < -3 | rstandard(fit) > 3)) ),"",rownames(housing) ),
     cex= 0.8,pos=2,col='red')
hist(hatvalues(fit))


plot(fit,which=2)


#########################################################################################################
#
# Residuals, Cooks Distance, Leverage ########  After Outlier being removed
#
#########################################################################################################
#-------------------------------------------Diagnostic Plot (Plot 1,3) #### Residuals
point_exclude <- c(187,215,365,366,368,369,370,371,372,373,413)
housing_remove <- housing[-(point_exclude),]
par(mfrow=c(2,2))
#rownames(housing_remove) = 1:nrow(housing_remove) #reset the rownames
fit_after <- lm (medv~crim+zn+indus+chas+nox+rm+age+dis+rad+tax+ptratio+b+lstat,data=housing_remove)
res_after <- fit_after$residuals
plot(fit_after,which=1)
text(predict(fit_after),res_after,ifelse( ((rownames(housing_remove)==375 | rownames(housing_remove)==162 | rownames(housing_remove)==408) 
                               | (!(res_after < -9 | res_after > 10)) ),"",rownames(housing_remove) ),
     cex= 0.8,pos=3,col='red')
#hist(res_after)

rstandards_after = rstandard(fit_after)
plot(predict(fit_after),rstandards_after,xlab="Fitted Values",ylab="Standardized Residuals",main="Standardize Residuals Vs Fitted")
text(predict(fit_after),rstandard(fit_after),ifelse((!(rstandards_after < -3 | rstandards_after > 3)),"",rownames(housing_remove) ),
     cex= 0.8,pos=2,col='red')
#hist(rstandards_after)

#-------------------------------------------Diagnostic Plot (Plot 4,5) #### Cooks Distaince, Leverage
#par(mfrow=c(2,2))
plot(fit_after,which=4)
text(rownames(housing_remove),cooks.distance(fit_after),ifelse( ((rownames(housing_remove)==366 | rownames(housing_remove)==372 | rownames(housing_remove)==405) 
                                                    | (cooks.distance(fit_after) < 0.03) ),"",rownames(housing_remove) ),
     cex= 0.8,pos=3,col='red')
#hist(cooks.distance(fit_after))
plot(fit_after,which=5)
text(hatvalues(fit_after),(rstandard(fit_after)),ifelse( ((rownames(housing_remove)==375 | rownames(housing_remove)==415) 
                                              | (!(rstandard(fit_after) < -3 | rstandard(fit_after) > 3)) ),"",rownames(housing_remove) ),
     cex= 0.8,pos=2,col='red')
#hist(hatvalues(fit_after))


par(mfrow=c(2,2))
plot(fit_after,which=2)
plot(fit_after,which=3)
text(predict(fit_after),sqrt(rstandard(fit_after)),ifelse( ((rownames(housing_remove)==375 | rownames(housing_remove)==408 | rownames(housing_remove)==162) 
                                                          | (!(sqrt(rstandard(fit_after)) < -1.8 | sqrt(rstandard(fit_after)) > 1.8)) ),"",rownames(housing_remove) ),
     cex= 0.8,pos=2,col='red')



#########################################################################################################
#
# Box Cox Transformation (Before Outlier is Removed)
#
#########################################################################################################
library(MASS)
bc = boxcox(fit)
bc = boxcox(fit,lambda = seq(-3,3))
best_lam=bc$x[which((bc$y == max(bc$y)))]
fit_after_boxcox <- lm ((((medv^best_lam)-1)/best_lam)~crim+zn+indus+chas+nox+rm+age+dis+rad+tax+ptratio+b+lstat,data=housing)
par(mfrow=c(2,2))
plot(fit_after_boxcox)



#########################################################################################################
#
# Box Cox Transformation (After Removing Outlier)
#
#########################################################################################################
#library(MASS)
bc = boxcox(fit_after,lambda = seq(-3,3))
best_lam=bc$x[which((bc$y == max(bc$y)))]
fit_modified_after_boxcox <- lm((((medv^best_lam)-1)/best_lam)~crim+zn+indus+chas+nox+rm+age+dis+rad+tax+ptratio+b+lstat,data=housing_remove)
par(mfrow=c(2,2))
plot(fit_modified_after_boxcox)


#########################################################################################################
#
# Plotting the Data (Original Vs Precited)
#
#########################################################################################################
xlim=c(0,50)
ylim=c(-4,4)
#plot(predict(fit),rstandard(fit),xlim=xlim,ylim=ylim,xlab="Fitted Values",ylab="Standardized Residuals",main="Standardize Residuals Vs Fitted")
#plot(predict(fit_after),rstandard(fit_after),xlab="Fitted Values",ylab="Standardized Residuals",main="Standardize Residuals Vs Fitted",xlim=xlim,ylim=ylim)
#plot((predict(fit_modified_after_boxcox))^(1/best_lam),rstandard(fit_modified_after_boxcox),xlab="Fitted Values",ylab="Standardized Residuals",main="Standardize Residuals Vs Fitted",xlim=xlim,ylim=ylim)
plot ((1+(predict(fit_modified_after_boxcox))*best_lam)^(1/best_lam),rstandard(fit_modified_after_boxcox),xlab="Fitted Values",ylab="Standardized Residuals",main="Standardize Residuals Vs Fitted",xlim=xlim,ylim=ylim)

par(mfrow=c(2,2))
y_pred <- predict(fit)
plot(housing$medv,y_pred,main = "Original Data")

y_pred <- predict(fit_after)
plot(housing_remove$medv,y_pred,main = "Modified Data (After Removing Outlier)")

y_pred <- predict(fit_after_boxcox)
plot(housing$medv,y_pred,main = "Original Data - BoxCox")

y_pred <- predict(fit_modified_after_boxcox)
plot((housing_remove$medv),y_pred,main = "Modified Data (After Removing Outlier) - Box Cox (No Reverse)")

y_pred <- predict(fit_modified_after_boxcox)
plot(housing_remove$medv,(1+(y_pred*best_lam))^(1/best_lam),xlab = "y_true (Original)",ylab="y_pred(Prediction)",main = "Modified Data (After Removing Outlier) - Box Cox (Reverse)")


