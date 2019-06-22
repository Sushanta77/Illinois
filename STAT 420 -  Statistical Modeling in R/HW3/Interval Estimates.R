#Loading and Cleaning the Dataset
autompg = read.table("https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data",stringsAsFactors = FALSE)
#View(autompg)
colnames(autompg) = c("mpg","cyl","disp","hp","wt","acc","year","origin","name")
autompg = subset(autompg,autompg$hp != "?")
autompg = subset(autompg,autompg$name != "plymouth reliant")
rownames(autompg) = paste(autompg$cyl,"cylinder",autompg$year,autompg$name)
autompg = subset(autompg,select=c("mpg","cyl","disp","hp","wt","acc","year"))
autompg$hp = as.numeric(autompg$hp)

auto_model = lm(mpg~wt+year,data=autompg)

null_model = lm(mpg~1,data=autompg)
full_model = lm(mpg~wt+year,data=autompg)
anova(null_model,full_model)
summary(full_model)

SSTot =  sum((autompg$mpg - mean(autompg$mpg)) ^ 2) / (nrow(autompg) - 1)
SSReg =  sum((predict(full_model) - mean(autompg$mpg)) ^ 2) / (3 - 1)
SSErr =  sum((autompg$mpg - predict(full_model))^2) / (nrow(autompg) - 3)


F = SSReg / SSErr
F



null_model = lm(mpg~wt+year,data=autompg)
full_model = lm(mpg~.,data=autompg)
anova(null_model,full_model)
summary(full_model)

SSTot =  sum((autompg$mpg - predict(null_model)) ^ 2) / (nrow(autompg) - 3)
SSReg =  sum((predict(full_model) - predict(null_model)) ^ 2) / (7 - 3)
SSErr =  sum((autompg$mpg - predict(full_model))^2) / (nrow(autompg) - 7)


F = SSReg / SSErr
F
