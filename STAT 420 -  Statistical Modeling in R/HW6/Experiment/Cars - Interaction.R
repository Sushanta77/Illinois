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

