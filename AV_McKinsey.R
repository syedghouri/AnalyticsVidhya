getwd()
setwd("E:/AV/McKinsey_Prediction")

library(dummies)
library(xgboost)
library(rBayesianOptimization)

data <- read.csv("train.csv")
str(data)

numeric_types <- unlist(lapply(data, is.numeric))  

data_numeric <- data[ , numeric_types]
data_cat <- data[,!numeric_types]

sum(is.na(data_numeric))
sum(is.na(data_cat))

for(i in 1:ncol(data_numeric)){
  data_numeric[is.na(data_numeric[,i]), i] <- mean(data_numeric[,i], na.rm = TRUE)
}

data_cat <- dummy.data.frame(data_cat)
data_complete <- cbind(data_numeric,data_cat)
str(data_complete)

dtrain <- xgb.DMatrix(as.matrix(data_complete[,-c(1,11)]), label = data_complete$renewal)

cv_folds <- KFold(
  data_complete$renewal
  , nfolds = 5
  , stratified = TRUE
  , seed = 5000)

xgb_cv_bayes <- function(eta, max.depth, min_child_weight, subsample,colsample_bytree,nround ) {
  cv <- xgb.cv(params = list(booster = "gbtree"
                             # , eta = 0.01
                             , eta = eta
                             , max_depth = max.depth
                             , min_child_weight = min_child_weight
                             , colsample_bytree = colsample_bytree
                             , subsample = subsample
                             #, colsample_bytree = 0.3
                             , lambda = 1
                             , alpha = 0
                             , objective = "binary:logistic"
                             , eval_metric = "rmse")
               , data = dtrain
               , nround = nround
               , folds = cv_folds
               , prediction = TRUE
               , showsd = TRUE
               , early_stopping_rounds = 10
               , maximize = TRUE
               , verbose = 0
               , finalize = TRUE,label=data_complete[,c(11)])
  list(Score = cv$evaluation_log[,min(test_rmse_mean)]
       ,Pred = cv$pred
       , cb.print.evaluation(period = 1))
}


cat("Calculating Bayesian Optimum Parameters\n")

OPT_Res <- BayesianOptimization(xgb_cv_bayes
                                , bounds = list(
                                  eta = c(0.001, 0.4)
                                  , max.depth = c(3L, 8L)
                                  , min_child_weight = c(0.1, 1)
                                  , subsample = c(0.3, 0.6)
                                  , colsample_bytree = c(0.6, 1),nround = c(600,1500) )
                                , init_grid_dt = NULL
                                , init_points = 10
                                , n_iter = 5
                                , acq = "poi"
                                
                                
                                , verbose = TRUE)


trees_model <- xgboost(dtrain,label = data_complete$renewal,params = list(objective='binary:logistic',
                                    eta=OPT_Res$Best_Par["eta"],
                                    max.depth = OPT_Res$Best_Par["max.depth"],
                                    min_child_weight = OPT_Res$Best_Par["min_child_weight"],
                                    subsample = OPT_Res$Best_Par["subsample"],
                                    colsample_bytree = OPT_Res$Best_Par["colsample_bytree"]
                                    ),nrounds = OPT_Res$Best_Par["nround"])


data_pred <- read.csv("test.csv")
str(data_pred)

numeric_types <- unlist(lapply(data_pred, is.numeric))  

data_pred_numeric <- data_pred[ , numeric_types]
data_pred_cat <- data_pred[,!numeric_types]


for(i in 1:ncol(data_numeric)){
  data_numeric[is.na(data_numeric[,i]), i] <- mean(data_numeric[,i], na.rm = TRUE)
}

data_pred_cat <- dummy.data.frame(data_pred_cat)
data_pred_complete <- cbind(data_pred_numeric,data_pred_cat)
str(data_pred_complete)

y_pred <- predict(trees_model,newdata = as.matrix(data_pred_complete[,-c(1)]))
write.table(cbind(data_pred_complete[,c(1)],y_pred),col.names = c("ID","Renewal"),file="predictions.csv",row.names = F,sep = ",")

