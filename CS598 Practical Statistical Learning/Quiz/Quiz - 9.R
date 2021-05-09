library(e1071)
spam = read.table(file="https://web.stanford.edu/~hastie/ElemStatLearn/datasets/spam.data")
names(spam)[58] = "Y"
spam$Y = as.factor(spam$Y)
testID = c(1:100, 1901:1960)
spam.test=spam[testID, ]; 
spam.train=spam[-testID, ];


##------------------------------------------------ 
## Linear SVM  
##------------------------------------------------
#Training Error (Cost = 1)
svmfit=svm(Y ~., kernel="linear", data=spam.train, cost=1)
summary(svmfit)
table(spam.train$Y, svmfit$fitted)
179 + 112
svmpred=predict(svmfit, newdata=spam.test)
table(spam.test$Y, svmpred)
12+2

#Training Error (Cost = 10)
svmfit=svm(Y ~., kernel="linear", data=spam.train, cost=10)
summary(svmfit)
table(spam.train$Y, svmfit$fitted)
178 + 110
svmpred=predict(svmfit, newdata=spam.test)
table(spam.test$Y, svmpred)
11 + 3

#Training Error (Cost = 50)
svmfit=svm(Y ~., kernel="linear", data=spam.train, cost=50)
summary(svmfit)
table(spam.train$Y, svmfit$fitted)
179 + 112
svmpred=predict(svmfit, newdata=spam.test)
table(spam.test$Y, svmpred)
11 + 3



##------------------------------------------------ 
#Gaussian kernel SVM  
##------------------------------------------------
#Training Error (Cost = 1)
svmfit=svm(Y ~., data=spam.train, cost=1)
summary(svmfit)
table(spam.train$Y, svmfit$fitted)
147 + 85
svmpred=predict(svmfit, newdata=spam.test)
table(spam.test$Y, svmpred)
10 + 4


#Training Error (Cost = 10)
svmfit=svm(Y ~., data=spam.train, cost=10)
summary(svmfit)
table(spam.train$Y, svmfit$fitted)
99 + 47

svmpred=predict(svmfit, newdata=spam.test)
table(spam.test$Y, svmpred)
11 + 4

#Training Error (Cost = 50)
svmfit=svm(Y ~., data=spam.train, cost=50)
summary(svmfit)
table(spam.train$Y, svmfit$fitted)
63 + 27

svmpred=predict(svmfit, newdata=spam.test)
table(spam.test$Y, svmpred)
14 + 3

