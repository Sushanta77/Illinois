#-----------------------------------------------------------------
# Boston Housting Data Solution using glmnet
#
#
#-----------------------------------------------------------------
#Captured the Data
myData = Boston 
names(myData)[14] = "Y"
iLog = c(1,3,5,6,8,9,10,14);
myData[,iLog] = log(myData[,iLog]);
myData[,2] = myData[,2,]/10;
myData[,7] = myData[,7]^2.5/10^4;
myData[,11] = exp(0.4 * myData[,11]) / 1000;
myData[,12] = myData[,12]/100;
myData[,13] = sqrt(myData[,13]);

#Move the last column of myData, the response Y, to the 1st column
myData = data.frame(Y = myData[,14], myData[,-14])
names(myData)[1] = "Y";
names(myData)

#Assign the Variable with size, features, X & y
n = dim(myData)[1];
p = dim(myData)[2]-1;
X = as.matrix(myData[,-1]);
Y = myData[,1]

#Split the data into 80% Training, 20% Test
ntest = round(n*0.2)
ntrain = n - ntest;
test.id = sample(1:n,ntest)
Ytest = myData[test.id,1]

#Fit the Full Model
full.mode = lm (Y ~ ., data = myData[-test.id,]);
Ytest.pred = predict(full.model, newdata = myData[test.id,])

sum((Ytest = Ytest.pred)^2)/ntest

myridge = glmnet(X[-test.id,],Y[-test.id],alpha = 0) #alpha = 0, means it's Ridge
plot(myridge, label = TRUE, xvar = "lambda")

myridge

summary(myridge)

coef(myridge)
