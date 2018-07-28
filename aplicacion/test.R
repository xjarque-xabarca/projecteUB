# data[,ncol(iris)]
# iris$Species

#seeds_wheat.csv
#HTRU_2.csv

source("load-libraries.R")

mydataset <- loadCSV("/opt/datascience_info/projecteUB/data/seeds_wheat.csv")
head(mydataset)


classifiers = c( "Linear SVC",  "KNeighbors", "RF-Bagging", "RandomForest", "AdaBoost", "DecisionTree", "ExtraTrees", "Ridge", "SGD" )  
# classifiers = c( "" )  

scores <- compare_classifiers(mydataset, classifiers)
scores
  


# paste(c("Bar plot of",colnames(mydataset)[ncol(mydataset)]), collapse=" ")

sapply(as.factor(myclasses), typeof)

mydataset <- iris
myclasses <- mydataset[ncol(mydataset)]
myclassesnames <- colnames(mydataset)[ncol(mydataset)]
myclassesnames
mytitle <- paste(c("Diagrama de barras de" , myclassesnames, collapse=" "))
barplot(table(myclasses), col="blue", xlab=myclassesnames, ylab="Cantidad", main=mytitle)


library(ggplot2)

df <- data.frame(dose=c("D0.5", "D1", "D2"),
                 len=c(4.2, 10, 29.5))
head(df)

p

myclasses <- mydataset[ncol(scores)]
typeof(myclasses)


# Basic barplot

x$name <- factor(x$name, levels = x$name[order(x$val)])
x$name  # notice the changed order of factor levels

scores$Model <- factor(scores$Model, levels = scores$Model[order(scores$"Mean Val. Accuracy")])
scores$Model  # notice the changed order of factor levels

myAccuracy <- as.numeric(scores$"Mean Val. Accuracy")
p<-ggplot(data=scores, aes(x=Model, y=myAccuracy)) + geom_bar(stat="identity")

# Horizontal bar plot
# p + scale_y_continuous(limits = c(0.75, 1), breaks = c(0.8, 0.9, 1.0)) + coord_flip()
p + coord_flip()


