## Question 3 ----

# load libraries and import data ----
library(caret)
library(pROC)
library(ggplot2)
library(rstudioapi)
library(tidyr)
library(class)

thyroid <- read.table("newthyroid.txt", header = TRUE, sep = ',')

# check the data
str(thyroid)
table(thyroid$class)
ratio.imbalance <- table(thyroid$class)[2]/table(thyroid$class)[1]   # calculate the ratio of imbalance
ratio.imbalance

# pre-process the data to make the class factor
thyroid$class <- as.factor(thyroid$class)   # make the class factor

# explore the boxplot to explore the distribution of samples for each feature
featurePlot(x = thyroid[,-1],
            y = thyroid$class,
            plot = "box",
            layout = c(5, 1),
            scales = list(y = list(relation="free"),
                          x = list(rot = 90)),
            auto.key = list(columns = 2))

## Question 3(1) ----
# random splitting
tt <- 10   # repeat 10 times
set.seed(12)  
train.index <- createDataPartition(thyroid$class, p = 0.7, # create index for 70% train set
                                   times = tt, list = FALSE) 

# create two vectors to record AUC for both knn and lda
auc.knn <- vector('numeric', tt)
auc.lda <- vector('numeric', tt)

# create a data frame containing different k values to choose from
k.grid <- expand.grid(k=seq(3, 15, by=2))

# use kNN and LDA for classification
for (i in 1:tt){
  train.feature <- thyroid[train.index[,i], -1]   # training features
  train.label <- thyroid$class[train.index[,i]]   # training labels
  test.feature <- thyroid[-train.index[,i], -1]   # test features
  test.label <- thyroid$class[-train.index[,i]]   # test labels
  
  # kNN (tune k based on AUC)
  fitControl <- trainControl(method = 'repeatedcv',   # 5-fold CV
                             number = 5,   
                             repeats = 3,   # 3 repetitions
                             summaryFunction = twoClassSummary, # compute AUC, sensitivity, specificity etc.
                             classProbs = TRUE)  # get the probabilities in prediction
  # training process
  set.seed(5)
  knn.fit <- train(train.feature, train.label, method = 'knn',
                   trControl = fitControl,
                   metric = 'ROC',  # tune k value based on AUC
                   preProcess = c('center', 'scale'),   # define pre-processing of the data
                   tuneGrid = k.grid)   # tuning values for k
  # test process
  prob.knn <- predict(knn.fit, test.feature, type = 'prob')
  # record AUC
  roc.knn <- roc(predictor = prob.knn$n, response = test.label) 
  auc.knn[i] <- roc.knn$auc
  
  # fit the LDA model
  lda.fit <- train(train.feature, train.label, method = "lda",
                   trControl = trainControl(method = "none"))
  # test process
  prob.lda <- predict(lda.fit, test.feature, type = 'prob')
  # record AUC
  roc.lda <- roc(predictor = prob.lda$n, response = test.label)
  auc.lda[i] <- roc.lda$auc
  
  # extract the ROC curve for first random splitting
  if (i==1){
    roc.knn1 <- roc.knn
    roc.lda1 <- roc.lda
    knn.fit1 <- knn.fit
    lda.fit1 <- lda.fit
    knn.pred1 <- predict(knn.fit, test.feature)
    lda.pred1 <- predict(lda.fit, test.feature)
  }
}

#View the confusion matrix to determine the classification accuracy and sensitivity
confusionMatrix(knn.pred1, test.label) #For the kNN method
confusionMatrix(lda.pred1, test.label) #For the LDA method

# view the AUC on the test set using classifier kNN and LDA
auc.knn
auc.lda

# Model assessment using the ROC curve for the random splitting
plot(roc.knn1, main = 'ROC Curve')
plot(roc.lda1, col = 'orange', add = TRUE)  
# add legend
legend("bottomright", legend=c("kNN","LDA"),
       col=c("skyblue","orange"), lty=c(1,1), cex=1, text.font=2)

## Question 3(2) ----
# Create a matrix of the 10 AUC values for kNN and LDA
auc_matrix <- cbind(auc.knn, auc.lda)
auc_matrix

# Create a vector to identify the methods
methods <- c(rep("kNN", 10), rep("LDA", 10))
methods

# Combine the AUC values and methods into a data frame
auc_df <- data.frame(AUC = c(auc.knn, auc.lda), Method = methods)
auc_df

#Creating the boxplots
boxplot<-ggplot(auc_df, aes(x=Method, y=AUC, fill=Method)) +
  geom_boxplot()
boxplot

## Question 3(3) ----

## tuning k based on sensitivity metric
# for the first training/test split
train.feature <- thyroid[train.index[,1], -1]   # training features
train.label <- thyroid$class[train.index[,1]]   # training labels
test.feature <- thyroid[-train.index[,1], -1]   # test features
test.label <- thyroid$class[-train.index[,1]]   # test labels
# kNN (tune k based on sensitivity)
fitControl <- trainControl(method = 'repeatedcv',   # 5-fold CV
                           number = 5,   
                           repeats = 3,   # No repeats
                           summaryFunction = twoClassSummary, # compute AUC, sensitivity, specificity etc.
                           classProbs = TRUE)  # get the probabilities in prediction
# training process
set.seed(5)
knn.fit <- train(train.feature, train.label, method = 'knn',
                 trControl = fitControl,
                 metric = 'Sens',  # tune k value based on sensitivity
                 preProcess = c('center', 'scale'),   # define pre-processing of the data
                 tuneGrid = k.grid)   # tuning values
# test process
knn.pred <- predict(knn.fit, test.feature)
confusionMatrix(knn.pred, test.label) 

# get ROC
prob.knn <- predict(knn.fit, test.feature, type = 'prob')
# record AUC
roc.knn <- roc(predictor = prob.knn$n, response = test.label) 
auc.knn2 <- roc.knn$auc
auc.knn2