library(e1071)
spam = read.table(file="https://web.stanford.edu/~hastie/ElemStatLearn/datasets/spam.data")
names(spam)[58] = "Y"
spam$Y = as.factor(spam$Y)
testID = c(1:100, 1901:1960)
spam.test=spam[testID, ]; 
spam.train=spam[-testID, ];

## Linear SVM  
svmfit=svm(Y ~., kernel="linear", data=spam.train, cost=1)
summary(svmfit)
table(spam.train$Y, svmfit$fitted)
svmpred=predict(svmfit, newdata=spam.test)
table(spam.test$Y, svmpred)

## Gaussian kernel SVM  
svmfit=svm(Y ~., data=spam.train, cost=1)
summary(svmfit)
table(spam.train$Y, svmfit$fitted)
svmpred=predict(svmfit, newdata=spam.test)
table(spam.test$Y, svmpred)