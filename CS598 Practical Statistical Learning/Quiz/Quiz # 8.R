#Save the Caravan dataset, for use b Python
write.csv(Caravan,"/Users/sushanta/Documents/GitHub/Illinois/CS598 Practical Statistical Learning/Quiz/Caravan.csv")

#Loading the Library
library(randomForest)
library(ISLR)
library(pROC)


#Check the dataset
dim(Caravan)

#Extract the Data
X_test1 = Caravan[0:1000,0:86]
X_test = Caravan[0:1000,0:85]
y_train = X_train[,86]
y_test = Caravan[0:1000,86]
X_train = Caravan[1001:dim(Caravan)[1],]
X_train_Only = Caravan[1001:dim(Caravan)[1],0:85]
yCaravan[1001:dim(Caravan)[1],86]

#Convert the y data into 0, 1 from "Yes" / "No"
y_test_transformed = ifelse(y_test=="Yes",1,0)
y_train_transformed = ifelse(y_train=="Yes",1,0)


#Fit the Logistic Regression
glm.fit = glm(formula = Purchase ~ ., data = X_train, family = binomial)
glm.probs.train = predict(glm.fit,type="response")
glm.probs.test = predict(glm.fit,X_test,type="response")

glm.pred = rep("No",1000)
glm.pred[glm.probs.test > 0.25] = "Yes"

#Confusion Matrix
table (glm.pred,y_test)

roc(y_test_transformed, glm.probs.test) # 0.7407


#------------------------------------------------------------------------------------------------------------------------
# AIC
#------------------------------------------------------------------------------------------------------------------------


model_start = glm(Purchase ~ 1, data = X_train, family = binomial)
stepAIC = step(model_start, scope = Purchase ~ MOSTYPE + MAANTHUI + MGEMOMV + MGEMLEEF + MOSHOOFD + 
                 MGODRK + MGODPR + MGODOV + MGODGE + MRELGE + MRELSA + MRELOV + 
                 MFALLEEN + MFGEKIND + MFWEKIND + MOPLHOOG + MOPLMIDD + MOPLLAAG + 
                 MBERHOOG + MBERZELF + MBERBOER + MBERMIDD + MBERARBG + MBERARBO + 
                 MSKA + MSKB1 + MSKB2 + MSKC + MSKD + MHHUUR + MHKOOP + MAUT1 + 
                 MAUT2 + MAUT0 + MZFONDS + MZPART + MINKM30 + MINK3045 + MINK4575 + 
                 MINK7512 + MINK123M + MINKGEM + MKOOPKLA + PWAPART + PWABEDR + 
                 PWALAND + PPERSAUT + PBESAUT + PMOTSCO + PVRAAUT + PAANHANG + 
                 PTRACTOR + PWERKT + PBROM + PLEVEN + PPERSONG + PGEZONG + 
                 PWAOREG + PBRAND + PZEILPL + PPLEZIER + PFIETS + PINBOED + 
                 PBYSTAND + AWAPART + AWABEDR + AWALAND + APERSAUT + ABESAUT + 
                 AMOTSCO + AVRAAUT + AAANHANG + ATRACTOR + AWERKT + ABROM + 
                 ALEVEN + APERSONG + AGEZONG + AWAOREG + ABRAND + AZEILPL + 
                 APLEZIER + AFIETS + AINBOED + ABYSTAND, direction = "forward" )

summary(stepAIC)

glm.probs.testAIC = predict(stepAIC,X_test,type="response")

glm.pred.AIC = rep("No",1000)
glm.pred.AIC[glm.probs.testAIC > 0.25] = "Yes"

#Confusion Matrix
table (glm.pred.AIC,y_test)

roc(y_test_transformed, glm.probs.testAIC) # 0.7353


#------------------------------------------------------------------------------------------------------------------------
# BIC
#------------------------------------------------------------------------------------------------------------------------
n = dim(X_train)[1]
model_start = glm(Purchase ~ 1, data = X_train, family = binomial)
stepBIC = step(model_start, scope = Purchase ~ MOSTYPE + MAANTHUI + MGEMOMV + MGEMLEEF + MOSHOOFD + 
                 MGODRK + MGODPR + MGODOV + MGODGE + MRELGE + MRELSA + MRELOV + 
                 MFALLEEN + MFGEKIND + MFWEKIND + MOPLHOOG + MOPLMIDD + MOPLLAAG + 
                 MBERHOOG + MBERZELF + MBERBOER + MBERMIDD + MBERARBG + MBERARBO + 
                 MSKA + MSKB1 + MSKB2 + MSKC + MSKD + MHHUUR + MHKOOP + MAUT1 + 
                 MAUT2 + MAUT0 + MZFONDS + MZPART + MINKM30 + MINK3045 + MINK4575 + 
                 MINK7512 + MINK123M + MINKGEM + MKOOPKLA + PWAPART + PWABEDR + 
                 PWALAND + PPERSAUT + PBESAUT + PMOTSCO + PVRAAUT + PAANHANG + 
                 PTRACTOR + PWERKT + PBROM + PLEVEN + PPERSONG + PGEZONG + 
                 PWAOREG + PBRAND + PZEILPL + PPLEZIER + PFIETS + PINBOED + 
                 PBYSTAND + AWAPART + AWABEDR + AWALAND + APERSAUT + ABESAUT + 
                 AMOTSCO + AVRAAUT + AAANHANG + ATRACTOR + AWERKT + ABROM + 
                 ALEVEN + APERSONG + AGEZONG + AWAOREG + ABRAND + AZEILPL + 
                 APLEZIER + AFIETS + AINBOED + ABYSTAND, direction = "forward", k=log(n))
summary(stepBIC)

glm.probs.testBIC = predict(stepBIC,X_test,type="response")

glm.pred.BIC = rep("No",1000)
glm.pred.BIC[glm.probs.testBIC > 0.25] = "Yes"

#Confusion Matrix
table (glm.pred.BIC,y_test)

roc(y_test_transformed, glm.probs.testBIC) # 0.7414



#------------------------------------------------------------------------------------------------------------------------
# glmnet
#------------------------------------------------------------------------------------------------------------------------
X_train = Caravan[1001:dim(Caravan)[1],]
x = model.matrix(Purchase~.,X_train)[,-1]
y = X_train$Purchase
x_test = model.matrix(Purchase~.,X_test1)[,-1]
y_transformed = ifelse(y=="Yes",1,0)
glmnet_model = glmnet(x,y_transformed,standardize = TRUE,intercept=TRUE )

summary(glmnet_model)

coef(glmnet_model)
dim(coef(glmnet_model))

glmnet.probs = predict(glmnet_model, s=0.004,newx=x_test,type="response")

glmnet.pred = rep("No",1000)
glmnet.pred[glmnet.pred > 0.25] = "Yes"

#Confusion Matrix
table (glmnet.pred,y_test)

roc(y_test_transformed, glmnet.probs) # 0.7501


#------------------------------------------------------------------------------------------------------------------------
# glmnet (One more time) - **** REVISED *****
#------------------------------------------------------------------------------------------------------------------------
X_test = Caravan[0:1000,0:85]
y_test = Caravan[0:1000,86]
X_train = Caravan[1001:dim(Caravan)[1],0:85]
y_train = X_train[,86]

myLasso2 <- glmnet(X_train,y_train,alpha=1,family = 'binomial',standardize = TRUE,intercept = TRUE)
coef(myLasso2,s=0.004)

glmnet.probs1 = predict(myLasso2, s=0.004,newx=x_test,type="response")
glmnet.probs1
glmnet.pred = rep("No",1000)
glmnet.pred[glmnet.probs1 > 0.25] = "Yes"
table (glmnet.pred,y_test)

roc(y_test_transformed, glmnet.probs1) # 0.7501



#------------------------------------------------------------------------------------------------------------------------
# glmnet (One more time)
#------------------------------------------------------------------------------------------------------------------------
X_test = Caravan[0:1000,0:85]
y_train = X_train[,86]
y_test = Caravan[0:1000,86]
X_train = Caravan[1001:dim(Caravan)[1],]
X_train_Only = Caravan[1001:dim(Caravan)[1],0:85]

y_test_transformed = ifelse(y_test=="Yes",1,0)
y_train_transformed = ifelse(y_train=="Yes",1,0)

myLasso2 <- glmnet(X_train_Only,y_train_transformed,alpha=1,family = 'binomial')
coef(myLasso2,s=0.004)

myLasso2 <- glmnet(X_train_Only,y_train,alpha=1,family = 'binomial')
coef(myLasso2,s=0.004)
glmnet.pred1 = rep("No",1000)
glmnet.pred1[glmnet.pred1 > 0.25] = "Yes"
table (glmnet.pred1,y_test)


glmnet_model1 = glmnet(X_train_Only,y_train_transformed,alpha=1,lambda=0.004,standardize = TRUE,intercept=TRUE)
coef(glmnet_model1)
glmnet.probs1 = predict(glmnet_model1, s=0.004,newx=x_test)


glmnet.pred1 = rep("No",1000)
glmnet.pred1[glmnet.pred1 > 0.25] = "Yes"

#Confusion Matrix
table (glmnet.pred1,y_test)

roc(y_test_transformed, glmnet.probs1) # 0.7501
