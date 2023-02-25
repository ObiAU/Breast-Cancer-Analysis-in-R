
#############################################################################################################
# Comment below out once run once!
install.packages("mlbench")
# Load the package
library(mlbench)

# Load the data
data(BreastCancer)

# Cleaning the data
# Class is response variable
# 9 characteristics are the predictor variables

# Check the size
dim(BreastCancer)

head(BreastCancer)
?BreastCancer

# First convert the response variables to binary 0 and 1

head(BreastCancer$Class)
typeof(BreastCancer$Class)
head(as.integer(BreastCancer$Class))
BreastCancer$Class

# Apply as numeric to the relevant columns referring to the predictor variables
BreastCancer1 <- apply(BreastCancer[c(2:10)], 2, as.numeric)
head(BreastCancer1)

# Recombine the Id and Class columns with the numerical form columns
BreastCancer2 <- data.frame(Id = BreastCancer$Id, BreastCancer1, Class = as.numeric(BreastCancer$Class)-1)
head(BreastCancer2)

# Apply the complete.cases tool to keep all rows that do not contain an NA value, thus removing all rows containing NA values.
MyBreastCancer <- BreastCancer2[complete.cases(BreastCancer2),]
head(MyBreastCancer)

# Test
complete.cases(MyBreastCancer)


# For our statistical computations, we do not require the Id column as it is not useful here.
MyBreastCancer_test <- MyBreastCancer[,2:11]
head(MyBreastCancer_test)

table(MyBreastCancer_test$Class)


# Summarize the data graphically and numerically.
# Discuss relationships between response and predictor variables
# Discuss relationships between predictor variables

# Install GGally package to extract ggpairs
# install.packages("GGally")
library(GGally)
library(ggplot2)
# Pairs plot of the predictors
pairs(MyBreastCancer_test)
cor(MyBreastCancer_test)
ggpairs(MyBreastCancer_test, axisLabels = c("none"),showStrips = TRUE, ggplot2::aes(colour = c('cadetblue1')))

# Logistic regression with best subset selection
# Fit a logistic regression model for the response variable (class)
# In terms of the predictor variables using best subset selection

# install.packages("bestglm")
library(leaps)
library(bestglm)

# We will perform best subset selection with respect to the BIC and AIC
# Our data is already such that the response variable column is the last column.
# We use the family equals binomial argument to indicate we want to fit a logistic reg model,
# and not some other kind of generalized linear model.

bss_fit_AIC <- bestglm(MyBreastCancer_test, family = binomial, IC = "AIC")
bss_fit_BIC <- bestglm(MyBreastCancer_test, family = binomial, IC = "BIC")

# Examine our results
bss_fit_AIC$Subsets
bss_fit_BIC$Subsets

# Find optimal number of predictors
(best_AIC <- bss_fit_AIC$ModelReport$Bestk)
(best_BIC <- bss_fit_BIC$ModelReport$Bestk)

# So it seems with respect to AIC the model should have 7 predictors,
# and for the lowest BIC, the model should have 5 predictors.

# For our best fitting models, we will estimate their test error using 10-fold cross-validation
# The function is given as
logistic_reg_bss_cv = function(X, y, fold_ind) {
  p = ncol(X)
  Xy = data.frame(X, y=y)
  X = as.matrix(X)
  nfolds = max(fold_ind)
  if(!all.equal(sort(unique(fold_ind)), 1:nfolds)) stop("Invalid fold partition.")
  fold_errors = matrix(NA, nfolds, p+1) # p+1 because M_0 included in the comparison
  for(fold in 1:nfolds) {
    # Using all *but* the fold as training data, find the best-fitting models 
    # with 0, 1, ..., p predictors, i.e. identify the predictors in M_0, M_1, ..., M_p
    tmp_fit = bestglm(Xy[fold_ind!=fold,], family=binomial, IC="AIC")
    best_models = as.matrix(tmp_fit$Subsets[,2:(1+p)])
    # Using the fold as test data, find the test error associated with each of 
    # M_0, M_1,..., M_p
    for(k in 1:(p+1)) {
      fold_errors[fold, k] = logistic_reg_fold_error(X[,best_models[k,]], y, fold_ind==fold)
    }
  }
  # Find the fold sizes
  fold_sizes = numeric(nfolds)
  for(fold in 1:nfolds) fold_sizes[fold] = length(which(fold_ind==fold))
  # For models with 0, 1, ..., p predictors compute the average test error across folds
  test_errors = numeric(p+1)
  for(k in 1:(p+1)) {
    test_errors[k] = weighted.mean(fold_errors[,k], w=fold_sizes)
  }
  # Return the test error for models with 0, 1, ..., p predictors
  return(test_errors)
}

# number of rows and columns
(n = nrow(MyBreastCancer_test))
(p = ncol(MyBreastCancer_test) - 1)

# We will set the random seed at 5 for reproducibility
set.seed(5)
# Create our k-fold sample
nfolds = 10
fold_index = sample(nfolds, n, replace=TRUE)
head(fold_index)

# We now apply the function to the best subset selection sets
# Note: we are still using MyBreastCancer_test
cv_errors = logistic_reg_bss_cv(MyBreastCancer_test[,1:p], MyBreastCancer_test[,p+1], fold_index)

(best_cv = which.min(cv_errors) - 1)

# Interestingly, the CV error criteria returns optimal number of predictors as 7.
# This is the same result as the AIC's.

# We can plot graphs analysing the optimal number of predictor variables with respect to the AIC, BIC,
# and CV error criteria as follows:

par(mfrow=c(2, 2))
MyBreastCancer_test$Class == MyBreastCancer_optimized$Class
# Plots that highlight the optimal number of predictor variables
plot(0:p, bss_fit_AIC$Subsets$AIC, xlab="Number of predictors", ylab="AIC", type="b")
points(best_AIC, bss_fit_AIC$Subsets$AIC[best_AIC+1], col="red", pch=16)
plot(0:p, bss_fit_BIC$Subsets$BIC, xlab="Number of predictors", ylab="BIC", type="b")
points(best_BIC, bss_fit_BIC$Subsets$BIC[best_BIC+1], col="red", pch=16)
plot(0:p, cv_errors, xlab="Number of predictors", ylab="CV error", type="b")
points(best_cv, cv_errors[best_cv+1], col="red", pch=16)

# The graphs display the optimal number of predictors as 7, 5 and 7 for the AIC, BIC, and CV error respectively.
# Thus we will build a reduced model with 7 predictor variables and compute its test error.

pstar = 7

# Identify which predictors are in the 7-predictor model
bss_fit_AIC$Subsets[pstar+1,]

# Thus our new, reduced model will not contain the Cell Size and Epith.c.size predictor variables.

# Build final model
(indices = which(bss_fit_AIC$Subsets[pstar+1, 2:(p+1)]==TRUE))

MyBreastCancer_optimized <- MyBreastCancer_test[,c(indices, p+1)]
MyBreastCancer_optimized

# Obtain the regression coefficients of this new model
logreg_fit = glm(MyBreastCancer_test$Class ~., data = MyBreastCancer_optimized, family = "binomial")
summary(logreg_fit)
logreg_fit$coefficients

(reduced_test_error = general_cv(MyBreastCancer_optimized[,1:pstar], MyBreastCancer_optimized[,pstar+1], fold_index, logistic_reg_fold_error))

# Classifiers for LDA and QDA on both reduced and non-reduced models
# First, we will write a function that normalizes a confusion matrix.
# It will compute the estimates of the classification probabilities by dividing each row
# y the row sum.
# We write the function as follows

NormaliseTool = function(x){
  return(x/sum(x))
}

# Now onto the LD and QD analyses.
library(MASS)

# Non-reduced model LDA

(lda_fit <- lda(MyBreastCancer_test$Class ~., data=MyBreastCancer_test))

# Now compute the training confusion matrix and training error

lda_predict = predict(lda_fit, MyBreastCancer_test)
yhat = lda_predict$class

# Confusion matrix:
(confusion = table(Observed=MyBreastCancer_test$Class, Predicted=yhat))
# Normalized
(confusion <- t(apply(confusion, 1, NormaliseTool)))

# Training error

(train_error = 1 - mean(MyBreastCancer_test$Class == yhat))


# Reduced model LDA
(lda_fit_red <- lda(MyBreastCancer_optimized$Class ~., data=MyBreastCancer_optimized))

# Now compute the training confusion matrix and training error
lda_predict_red = predict(lda_fit_red, MyBreastCancer_optimized)
yhat = lda_predict_red$class

# Confusion matrix for lda:
(confusion_lda_red = table(Observed=MyBreastCancer_optimized$Class, Predicted=yhat))
# Normalized
 t(apply(confusion_lda_red, 1, NormaliseTool))

# QDA for reduced model
(qda_fit_red <- qda(MyBreastCancer_optimized$Class ~., data=MyBreastCancer_optimized))

# Now compute the training confusion matrix and training error
qda_predict_red = predict(qda_fit_red, MyBreastCancer_optimized)
zhat = qda_predict_red$class
# Confusion matrix for qda:
(confusion_qda_red = table(Observed=MyBreastCancer_optimized$Class, Predicted=zhat))
# Normalized
t(apply(confusion_qda_red, 1, NormaliseTool))

(confusion_lda_red = table(Observed=MyBreastCancer_optimized$Class, Predicted=yhat))
(confusion_qda_red = table(Observed=MyBreastCancer_optimized$Class, Predicted=zhat))

lda_fit_red$means
qda_fit_red$means

t(apply(confusion_lda_red, 1, NormaliseTool))
t(apply(confusion_qda_red, 1, NormaliseTool))


# Training error

(train_error_red = 1 - mean(MyBreastCancer_optimized$Class == yhat))

# 10-fold cross-validation to estimate test error

# Function to compute test error for LDA given specified training and validation data
lda_fold_error = function(X, y, test_data){
  Xy = data.frame(X, y=y)
  if(ncol(Xy)>1) tmp_fit = lda(y ~ ., data=Xy[!test_data,])
  tmp_predict = predict(tmp_fit, Xy[test_data,])
  yhat = tmp_predict$class 
  yobs = y[test_data]
  test_error = 1 - mean(yobs == yhat)
  return(test_error)
}

# Estimated test error for lda performed on the full model

(test_error = general_cv(MyBreastCancer_test[,1:p], MyBreastCancer_test[,p+1], fold_index, lda_fold_error))

# Estimated test error for lda performed on the reduced model

(test_error_red = general_cv(MyBreastCancer_optimized[,1:pstar], MyBreastCancer_optimized[,pstar+1], fold_index, lda_fold_error))

# Slightly surprisingly, the test and training errors for both models with respect to lda are the exact same.


# QDA for full model
(qda_fit <- qda(MyBreastCancer_test$Class ~., data=MyBreastCancer_test))

# Now compute the training confusion matrix and training error

qda_predict = predict(qda_fit, MyBreastCancer_test)
yhat = qda_predict$class

# Confusion matrix:
(confusion2 = table(Observed=MyBreastCancer_test$Class, Predicted=yhat))
# Normalized
(confusion2 <- t(apply(confusion2, 1, NormaliseTool)))

# Training error

(train_error = 1 - mean(MyBreastCancer_test$Class == yhat))


# Reduced training error

(train_error_red = 1 - mean(MyBreastCancer_optimized$Class == yhat))


# QDA for the reduced model returns a higher train error than that of the full model QDA.
# QDA for the full model returns the exact same train error as the LDA reduced model train error.
# LDA for the full model returns the lowest training error out of all simulations we have ran.
# This suggests LDA is arguably a more accurate comparison metric for the model.


# Function to compute test error for QDA given specified training and validation data
qda_fold_error = function(X, y, test_data) {
  Xy = data.frame(X, y=y)
  if(ncol(Xy)>1) tmp_fit = qda(y ~ ., data=Xy[!test_data,])
  tmp_predict = predict(tmp_fit, Xy[test_data,])
  yhat = tmp_predict$class 
  yobs = y[test_data]
  test_error = 1 - mean(yobs == yhat)
  return(test_error)
}

# Full model test error

(test_error = general_cv(MyBreastCancer_test[,1:p], MyBreastCancer_test[,p+1], fold_index, qda_fold_error))


# Reduced model test error

(test_error_red = general_cv(MyBreastCancer_optimized[, 1:pstar],
                             MyBreastCancer_optimized[, pstar+1], fold_index, qda_fold_error))

# Both return a test error that is higher than their respective train errors
# It is not good when a model performs worse on it's test/validation data than its training data.
# QDA also returns a higher test error than LDA for both reduced and full models.
# This is further evidence that performing LDA here is more optimal.


# k-fold cross-validation
# We will use 10-fold cross validation as it is a widely considered, effective cross-validation
# methodology.

# Underlying functions to help us calculate test error
logistic_reg_fold_error = function(X, y, test_data) {
  Xy = data.frame(X, y=y)
  if(ncol(Xy)>1) tmp_fit = glm(y ~ ., data=Xy[!test_data,], family="binomial")
  else tmp_fit = glm(y ~ 1, data=Xy[!test_data,,drop=FALSE], family="binomial")
  phat = predict(tmp_fit, Xy[test_data,,drop=FALSE], type="response")
  yhat = ifelse(phat > 0.5, 1, 0) 
  yobs = y[test_data]
  test_error = 1 - mean(yobs == yhat)
  return(test_error)
}

general_cv = function(X, y, fold_ind, fold_error_function) {
  p = ncol(X)
  Xy = cbind(X, y=y)
  nfolds = max(fold_ind)
  if(!all.equal(sort(unique(fold_ind)), 1:nfolds)) stop("Invalid fold partition.")
  fold_errors = numeric(nfolds)
  # Compute the test error for each fold
  for(fold in 1:nfolds) {
    fold_errors[fold] = fold_error_function(X, y, fold_ind==fold)
  }
  # Find the fold sizes
  fold_sizes = numeric(nfolds)
  for(fold in 1:nfolds) fold_sizes[fold] = length(which(fold_ind==fold))
  # Compute the average test error across folds
  test_error = weighted.mean(fold_errors, w=fold_sizes)
  # Return the test error
  return(test_error)
}


# COMPLETE COMPARISONS

# Test errors
(test_error = general_cv(MyBreastCancer_test[,1:p], MyBreastCancer_test[,p+1], fold_index, logistic_reg_fold_error))
(reduced_test_error = general_cv(MyBreastCancer_optimized[,1:pstar], MyBreastCancer_optimized[,pstar+1], fold_index, logistic_reg_fold_error))
# Estimated test error for lda performed on the fitted model
(test_error_lda = general_cv(MyBreastCancer_optimized[,1:pstar], MyBreastCancer_optimized[,pstar+1], fold_index, lda_fold_error))
# Estimated test error for qda performed on the fitted model
(test_error_qda = general_cv(MyBreastCancer_optimized[, 1:pstar], MyBreastCancer_optimized[, pstar+1], fold_index, qda_fold_error))



# Missing values is a common problem faced by a data set. There are several ways to overcome missing values
# such as omitting the observations, replacing the missing values with mean/mode 



#######################################################################################
# DELETED CODE


# Benign is 1 and malignant is 2
# We will change these to Benign = 0 and Malignant = 1 for convenience.
# 
# DataCleaningTool = function(df){
#   data <- data[complete.cases(data),]
#   data <- as.data.frame(lapply(df, as.numeric))
#   return(data)}
# 
# # Index out Id column and convert final (class) column to binary form before data cleaning
# BreastCancer1 <- DataCleaningTool(BreastCancer)
# BreastCancer2 <- data.frame(BreastCancer1[, -11], class=as.integer(BreastCancer1$Class)-1)
# 
# # Apply data cleaning tool
# MyBreastCancer <- data.frame(BreastCancer2[, -1])
# 
# MyBreastCancer

# Change final column (response variable) to binary form
# BreastCancer1 <- as.data.frame(lapply(BreastCancer, as.numeric))
# BreastCancer2 <- BreastCancer1[complete.cases(BreastCancer1),]
# BreastCancer3 <- data.frame(BreastCancer2[, -11], class=as.integer(BreastCancer2$Class)-1)


# Remove column representing patient ID as it is not relevant here
# MyBreastCancer <- data.frame(BreastCancer3[, -1])
