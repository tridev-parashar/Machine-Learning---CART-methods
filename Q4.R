## Question 4 ----

#Load libraries and the german credit dataset
library(caret)
GermanCredit<-read.csv("GermanCredit.csv")

# Delete two variables where all values are the same for both classes
GermanCredit[,c("Purpose.Vacation","Personal.Female.Single")] <- list(NULL)

#Get the training and test dataset
set.seed(12)
trainIndex <- createDataPartition(GermanCredit$Class, p = 0.7, list = FALSE, times = 1)
train_feature<- GermanCredit[trainIndex,-10] #training features
test_feature<- GermanCredit[-trainIndex,-10] # test features
train_label<-ifelse(GermanCredit[trainIndex,10] == "Bad", -1, 1)
test_label<-ifelse(GermanCredit[-trainIndex,10] == "Bad", -1, 1)

# Create a function to standardize the data
standardize <- function(data) {
  data <- scale(data, center = TRUE, scale = TRUE)
  data <- apply(data, MARGIN = 2, FUN = function(x) (x - min(x))/(max(x) - min(x)))
  return(data)
}

# Standardize the training and testing datasets
train_feature <- standardize(train_feature)
test_feature <- standardize(test_feature)

#Defining the function for a local fisher discriminant analysis
myFDA <- function(X,y){
  ########################################################
  # This function calculates the linear discriminant for binary
  # classification.
  # Input: Feature matrix, X (N by p) and label vector, y (N by 1)
  # Output: Linear discriminant, w (p by 1)
  ########################################################
  k <- 2   # for binary classification task
  n <- nrow(X)   # total number of observations
  # get class names
  name1 <- levels(factor(y))[1]
  name2 <- levels(factor(y))[2]
  # boolean values for each class
  bool.k1 <- (y == name1)
  bool.k2 <- (y == name2)
  
  # divide features into 2 groups according to labels
  class1 <- X[bool.k1,]
  class2 <- X[bool.k2,]
  # number of observations within each class
  n1 <- nrow(class1)
  n2 <- nrow(class2)
  # get covariance matrix for each group
  cov1 <- cov(class1)
  cov2 <- cov(class2)
  
  # get the within-class scatter
  sigma <- 1/(n-k) * (cov1*(n1-1) + cov2*(n2-1))   # using covariance matrix
  sigma.inv <- solve(sigma)   # inverse of matrix
  
  # calculate mean difference between classes for each feature
  mu <- data.frame(mu1 = matrix(apply(class1, 2, mean)),   # feature means of class 1
                   mu2 = matrix(apply(class2, 2, mean)))   # feature means of class 2
  mu.diff <- matrix(mu[,2] - mu[,1])
  
  # (mu2-mu1) is normalised by the within-class scatter
  coef <- sigma.inv %*% mu.diff
  scalar <- sqrt(t(coef) %*% sigma %*% coef)
  w <- coef/drop(scalar)  # get the vector of discriminant coefficients
  
  return(w)
}

#Applying the linear discriminant from the myFDA function in training and test sets
coef_training <- myFDA(X=train_feature,y=train_label)
coef_test <- myFDA(X=test_feature,y=test_label)

#Since the credit dataset in question has a singular matrix, one possibility is that the features
#may be multicollinear. Hence, we need to check for multicollinearity 

library(corrplot)
correlations <- cor(GermanCredit[,-10],method="pearson")
corrplot(correlations, number.cex = .9, method = "circle", 
         type = "full", tl.cex=0.30,tl.col = "black") #Plot for correlation matrix

#As observed, there is significant correlation between the features
#Therefore, we will be using dimension reduction methods (PCA) to overcome the
#challenge of singular matrix, which is arising due to multicollinearity

# We will now look at the PCA function prcomp() which automatically calculates these principal
#components.
# Running PCA on the training dataset
pca_model <- prcomp(train_feature, center = TRUE, scale. = TRUE)
# Plot the variance explained by each principal component
plot(pca_model)

#Proportion of variance plots for PCA data for the training set
pov_train <- cumsum(pca_model$sdev^2/sum(pca_model$sdev^2))
plot(pov_train, xlab = "Principal Component", ylab ="
Cumulative Proportion of Variance Explained", ylim = c(0,1),
     type = "b")

#Multiply the loading of the principal component to the training dataset
train_data_pca <- as.matrix(prcomp(train_feature,rank.= 45)$x)
test_data_pca <- as.matrix(prcomp(test_feature,rank.= 45)$x)

#Applying the linear discriminant from the myFDA function in PCA training data sets
fda_model <- myFDA(train_data_pca,train_label)

# Make predictions on the testing dataset using the FDA model
test_pred1 <- sign(test_data_pca %*% fda_model)

# Calculate the accuracy of the predictions
accuracy <- sum(test_pred1 == test_label) / length(test_label)

# Print the accuracy
cat("Accuracy:", accuracy, "\n") 

#We could further comment on the w achieved from the myFDA function by comparing it with 
#the LDA function (to check any similarity and improvement in accuracy)

coef1 <- as.vector(fda_model)
coef1

# fit the LDA model
lda_model <- lda(train_data_pca,train_label)
# get the linear discriminant using lda() function
coef2 <- lda_model$scaling   # coefficients of linear discriminants
coef2 <- as.vector(coef2)
coef2

#To check for any similarity between the two, we can check the cosine similarity between the two
cos <- t(coef1)%*%coef2 / (sqrt(t(coef1)%*%coef1) * sqrt(t(coef2)%*%coef2))
cos #A score of 1 suggests that the myFDA function is very much similar to LDA.

#Since both are similar, we could check the accuracy of the prediction using LDA functions
test_pred2 <- predict(lda_model, newdata = test_data_pca)$class
# Calculate the accuracy of the predictions
accuracy2 <- sum(test_pred2 == test_label) / length(test_label)
# Print the accuracy
cat("Accuracy:", accuracy2, "\n") #The accuracy under LDA has increase in comparison to FDA