---------------------------------------------------------------------------------------------------------
###Question - 1
beta_0 = -3
beta_1 = 1
beta_2 = 2 
beta_3 = 3

x1 = -1
x2 = 0.5
x3 = 0.25

eta = beta_0 + beta_1 * x1 + beta_2 * x2 + beta_3 * x3
p = 1/(1+exp(-eta))
1 - p
---------------------------------------------------------------------------------------------------------
  ###Question - 2
model = glm(am ~ mpg + hp + qsec, data = mtcars, family = binomial)
summary(model)$coefficient[4,1]
---------------------------------------------------------------------------------------------------------
###Question - 3
summary(model)$coefficient[2,1]
---------------------------------------------------------------------------------------------------------
###Question - 4  
model = glm(am ~ mpg + hp + qsec, data = mtcars, family = binomial)
beta_0 = summary(model)$coefficient[1,1]
beta_1 = summary(model)$coefficient[2,1]
beta_2 = summary(model)$coefficient[3,1]
beta_3 = summary(model)$coefficient[4,1]
  
newdata = data.frame(mpg = 19,hp = 150, qsec = 19)

beta_0 + beta_1 * 19 + beta_2 * 150 + beta_3 * 19

#predict(model, newdata = newdata, type = "response")
#1/(1+exp(-0.000239))

---------------------------------------------------------------------------------------------------------
###Question - 5  
newdata = data.frame(mpg = 22,hp = 123, qsec = 18)
eta = predict(model, newdata = newdata, type = "response")
eta

#Alternate way
eta = beta_0 + beta_1 * 22 + beta_2 * 123 + beta_3 * 18
p = 1/(1+exp(-eta))
p


---------------------------------------------------------------------------------------------------------
###Question - 6   (likeliood ratio test)
null_model = glm(am ~ 1, data = mtcars, family = binomial)
full_model = glm(am ~ mpg + hp + qsec, data = mtcars, family = binomial)
anova(null_model, full_model, test = "LRT")['Deviance'][,1][2]
---------------------------------------------------------------------------------------------------------
###Question - 7   (Wald Test)
null_model = glm(am ~ mpg + qsec, data = mtcars, family = binomial)
full_model = glm(am ~ mpg + hp + qsec, data = mtcars, family = binomial)
anova(null_model, full_model, test = "LRT")[2,'Pr(>Chi)']
---------------------------------------------------------------------------------------------------------
###Question - 8
Pima_tr = MASS::Pima.tr
Pima_te = MASS::Pima.te

model  = glm(type ~ glu + ped + I(glu ^ 2) + I(ped ^ 2) + glu:ped, data =  Pima_tr, family = binomial)
summary(model)
summary(model)$coefficient[5,1]
---------------------------------------------------------------------------------------------------------
###Question - 9 (Still Issues!!)
predict_response = predict(model, data = Pima_te, type = "response")
mean(predict_response > 0.8)*100
---------------------------------------------------------------------------------------------------------
###Question - 10 (AIC Backward)  
model  = glm(type ~ ., data =  Pima_tr, family = binomial)
model_back_aic = step(model, direction = "backward")  
---------------------------------------------------------------------------------------------------------
###Question - 11 (AIC Backward)    
model  = glm(type ~ . + .  ^ 2, data =  Pima_tr, family = binomial)
model_back_aic = step(model, direction = "backward")  
summary(model_back_aic)$'deviance'
---------------------------------------------------------------------------------------------------------
  
