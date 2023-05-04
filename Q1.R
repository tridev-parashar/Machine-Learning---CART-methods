## Question 1 ----

# Load libraries and import data ----
library(caret)
library(rattle)
library(rpart)
library(rpart.plot)
library(randomForest)
library(pROC)
library(ggplot2)
library(party)
GermanCredit<-read.csv("GermanCredit.csv")

# Data Pre-processing
str(GermanCredit)   # check variables type and structure
summary(GermanCredit)   # check statistics summary
attributes(GermanCredit$Class)   # check labels
table(GermanCredit$Class)   # check number of instances belonging to each label

# calculate the ratio of imbalance between the response classes
ratio.imbalance <- table(GermanCredit$Class)[2]/table(GermanCredit$Class)[1]   
ratio.imbalance

# check whether there are variables that have the same values for 2 classes
feature <- GermanCredit[,-10]
nfeature <- ncol(feature)
ll <- vector("numeric", nfeature)
for(ii in 1:nfeature){
  ll[ii] <- length(unique(feature[,ii])) # extract unique elements of each feature and record the number of unique elements
}
idx.uni <- which(ll==1)   # get the index of columns that have exactly the same values

# delete 2 variables that have the same values for both classes
feature[,idx.uni] <- list(NULL)

# random split
set.seed(12)   # for reproducibility
# create index for 70% training set
idx.train <- createDataPartition(GermanCredit$Class, 
                                 p = 0.7, times = 1, list = FALSE)
train.feature <- feature[idx.train,]   # training features
train.label <- GermanCredit$Class[idx.train]   # training labels
test.feature <- feature[-idx.train,]   # test features
test.label <- GermanCredit$Class[-idx.train]   # test labels
# Convert response variable to appropriate class
train.label <- as.factor(train.label)
test.label <- as.factor(test.label)

# combine into training set and test set
train <- cbind(train.feature, Class = train.label)
test <- cbind(test.feature, Class = test.label)

## 1) Decision Tree ----
# train decision trees using 5-fold cross-validation
set.seed(12)
dt <- rpart(Class ~., data = train, method = 'class',
            control = rpart.control(xval = 5,   # 5-fold cross-validation
                                    minbucket = 2,   # min number of obs in any terminal node
                                    cp = 0.01))   # threshold of complexity parameter
# print a table of fitted models
printcp(dt)
dt$cptable
# plot (relative) cv errors against tree size and cp
plotcp(dt)
# get the cp value which corresponds to min cv error
cp.best <- dt$cptable[which.min(dt$cptable[,'xerror']),'CP']
cp.best
# prune the decision tree using the best cp
dt.prune <- prune(dt, cp = cp.best)
# visualise the pruned tree
dt.plot <- rpart.plot(dt.prune, 
                      extra = 104, # show fitted class, probs, percentages
                      box.palette = "RdGn", # color scheme of boxes
                      branch.lty = 3, # dotted branch lines
                      shadow.col = "gray", # shadows under the node boxes
                      nn = TRUE, # display the node numbers
                      roundint =TRUE) # round int variable
# compute training error
pred.dt <- predict(dt.prune, train.feature, type = 'class')
err.train <- mean(pred.dt!=train.label)
err.train
# compute test error
pred.dt2 <- predict(dt.prune, test.feature, type = 'class')
err.test <- mean(pred.dt2!=test.label)
err.test

## 2) Random Forest ----
#Tuning random forest parameters i.e. determining the best number of predictors (mtry)
set.seed(12)
bestmtry <- tuneRF(train[, -which(names(train) == "Class")], train$Class,
                   stepFactor=1.5, plot=FALSE, cv.fold=5, ntree=1000)
bestmtry <- as.data.frame(bestmtry)
best_mtry <- bestmtry$mtry[which.min(bestmtry$OOBError)]
best_mtry

# producing random forest model 
set.seed(12)
rf <- randomForest(train$Class~.,data=train,
                   mtry=best_mtry,
                   importance=TRUE,
                   ntree=1000)
rf
# plot the random forest model
par(mfrow = c(1, 1))
plot(rf)
# compute test error
pred.rf <- predict(rf, test.feature)
err.test.rf <- mean(pred.rf!=test.label)
err.test.rf
# visualise the importance of features
varImpPlot(rf, cex = 0.6, main = "Variable Importance Plots")

## 3) ROC curves ----
# obtain the predicted probabilities associated with 2 classes from decision tree and 
#random forest models
prob.dt <- predict(dt.prune, test.feature, type = 'prob')
prob.rf <- predict(rf, test.feature, type = 'prob')
# build ROC curves
roc.dt <- roc(predictor = prob.dt[,1], response = test.label)
roc.rf <- roc(predictor = prob.rf[,1], response = test.label)
# get AUC values
auc.dt <- roc.dt$auc
auc.dt
auc.rf <- roc.rf$auc
auc.rf
# plot ROC curves
plot(roc.dt, main = 'ROC Curve')
plot(roc.rf, col = 'blue', add = TRUE)  
# add legend
legend("bottomright", legend=c("Decision Tree","Random Forest"),
       col=c("black","blue"), lty=c(1,1), cex=1, text.font=2)