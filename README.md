---
title: "Practical Machine Learning - Wearable Device Data Analysis"
author: "Xiang Cheng"
date: "October 24, 2015"
output: html_document
---
##Description:
Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement – a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, your goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. More information is available from the website here: <http://groupware.les.inf.puc-rio.br/har> (see the section on the Weight Lifting Exercise Dataset). 

##Data 
The training data for this project are available here: 
<https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv>


The test data are available here: 
<https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv>

The data for this project come from this source: <http://groupware.les.inf.puc-rio.br/har>. If you use the document you create for this class for any purpose please cite them as they have been very generous in allowing their data to be used for this kind of assignment. 

##Goal
The goal of this project is to predict the manner in which they did the exercise. In the datasets, it is the "classe" variable that we need to predict using machine learning methods.


##Data Analysis Steps
The general steps are:

1. Data preprossing: keep variables with at least 90% non-NA data points
2. Data slicing: using the training data set, split the data into trianing(60%) and testing (40%)
3. Train the Models: Test 5 methods (SVM, Random Forest, gradiant Stochastic Gradient Boosting, navie bayes, LDA)
4. Evaluate the model using the testing set
5. Finally test the model using the separate test set

####Step 1. Data preprocessing
Keep only variables with at least 90% non-NA data points. Also, because we need to finally test the model with the separate test set **_pml-testing.csv_**, we need to make sure all the variables we used in the training are available in the test set.
```{r, eval=FALSE}
trainingRaw = read.csv("pml-training.csv")
NAsTrain = apply(trainingRaw, 2, function(x) {sum(is.na(x))}) # get num of NAs of each variable
col2keep = (NAsTrain<0.1*nrow(trainingRaw)) # keep only variables with 90+% non-NAs
trainData = trainingRaw[, col2keep]  # good data for training 
trainData = trainData[ , -grep("timestamp|X|user_name|new_window",names(trainData))]
namesTrain = names(trainData)
dim(trainData)

finalTestRaw = read.csv("pml-testing.csv")
NAsTest = apply(finalTestRaw, 2, function(x) {sum(is.na(x))})
col2keep = NAsTest<0.1*nrow(finalTestRaw) 
finalTest = finalTestRaw[, col2keep] # good data for final testing
namesTest = names(finalTest)

colTrain2keep = sapply(namesTrain, function(x){x %in% namesTest}) # only keep variables in final test set
trainData = trainData[, colTrain2keep] # only keep variables availables in both training and test
trainData$classe = trainingRaw$classe # add back the target variable

```
What we got after step 1 are 2 separate datasets: **_trainingData_** for model training and test; **_finalTest_** for final model validation.

###Step 2. Data Slicing
Using only the training data set from **_pml-trainging.csv_**, split the data into trianing(60%) and testing (40%).

```{r, eval=FALSE}
# training data is splitted into training set and test set
inTrain = createDataPartition(trainData$classe, p=0.6, list=FALSE)
training = trainData[inTrain, ]
testing = trainData[-inTrain, ]
```

####Step 3. Train the Models
Train the models using 5 methods with 5-fold cross validation:

1. SVM
2. Random Forest
3. gradiant Stochastic Gradient Boosting
4. Navie Bayes
5. LDA

Those model are based different assumptions, such as linear relation, independences, etc, and could give us a wide range of insights of how the actual model may be. 

```{r, eval=FALSE}
ctrl = trainControl(method="cv", number=5)
models = c("svmLinear", "rf", "gbm", "nb", "lda") 

print(models[1])
modSVM = train(training$classe ~ ., data=training, method=models[1], trControl = ctrl)

print(models[2])
modRF = train(classe ~ ., data=training, method=models[2], trControl=ctrl, prox=T, ntree=200)

print(models[3])
modGBM = train(classe ~ ., data=training, method=models[3], trControl=ctrl, verbose=FALSE)

print(models[4])
modNB = train(classe ~ ., data=training, method=models[4], trControl=ctrl)

print(models[5])
modLDA = train(classe ~ ., data=training, method=models[5], trControl=ctrl)

```

####Step 4. Evaluate the model using the testing set
As we expected, the out of sample error using the testing set should be larger than that from the cross-validation. However, the out of sample error calculated below is more reliable to evaluate the models.

```{r, eval=FALSE}
predSVM = predict(modSVM, newdata=testing)
confusionMatrix(predSVM, testing$classe)

predRF = predict(modRF, newdata=testing)
confusionMatrix(predRF, testing$classe)
print("Variables importance in model")
vi = as.data.frame(varImp(modRF$finalModel))

predGBM = predict(modGBM, newdata=testing)
confusionMatrix(predRF, testing$classe)

predNB = predict(modNB, newdata=testing)
confusionMatrix(predNB, testing$classe)

predLDA = predict(modLDA, newdata=testing)
confusionMatrix(predLDA, testing$classe)
```
From the results, we found that random forest is the best model with the best performance in terms of sensitivity, specificity, and balanced accuracy. 


Moreover, from the results of random forest, we can check the importance of varialbes. The top 5 important variables are:

1. num_window           
2. roll_belt           
3. pitch_forearm          
4. magnet_dumbbell_y      
5. yaw_belt               
6. magnet_dumbbell_z      
7. pitch_belt             
8. roll_forearm           

```
# Output from confusionMatrix of random forest
Confusion Matrix and Statistics

          Reference
Prediction    A    B    C    D    E
         A 2231    1    0    0    0
         B    1 1516    2    0    0
         C    0    1 1366    7    0
         D    0    0    0 1278    6
         E    0    0    0    1 1436

Overall Statistics
                                          
               Accuracy : 0.9976          
                 95% CI : (0.9962, 0.9985)
    No Information Rate : 0.2845          
    P-Value [Acc > NIR] : < 2.2e-16       
                                          
                  Kappa : 0.9969          
 Mcnemar's Test P-Value : NA              

Statistics by Class:

                     Class: A Class: B Class: C Class: D Class: E
Sensitivity            0.9996   0.9987   0.9985   0.9938   0.9958
Specificity            0.9998   0.9995   0.9988   0.9991   0.9998
Pos Pred Value         0.9996   0.9980   0.9942   0.9953   0.9993
Neg Pred Value         0.9998   0.9997   0.9997   0.9988   0.9991
Prevalence             0.2845   0.1935   0.1744   0.1639   0.1838
Detection Rate         0.2843   0.1932   0.1741   0.1629   0.1830
Detection Prevalence   0.2845   0.1936   0.1751   0.1637   0.1832
Balanced Accuracy      0.9997   0.9991   0.9987   0.9964   0.9978
```
####Step 5. Finally test the model using the separate test set
The predictions is stored in a list.
```{r, eval=FALSE}
finalTestPred = list()
finalTestPred$SVM = predict(modSVM, newdata=finalTest)
finalTestPred$RF = predict(modRF, newdata=finalTest)
finalTestPred$GBM = predict(modGBM, newdata=finalTest)
finalTestPred$NB = predict(modNB, newdata=finalTest)
finalTestPred$LDA = predict(modLDA, newdata=finalTest)
```
For convenience of output and results comparison, we convert the list to a data frame and output the data frame to a file.
```{r, eval=FALSE}
# write predictions to file
finalTestPredDF = as.data.frame(finalTestPred)
write.table(finalTestPredDF, file="finalTestPred.csv", sep=",", row.names=FALSE)
```

##Summary
The exercise manner is predicted using 5 different machine learning algorithms. Of those 5 methods, the prediction performance is evaluated from the training set, testing set, and final test set. As we expected, the out of sample accuracy is slight worse than the error from the cross validation. Random forest is best method, and naive bayes and SVM perform the worst. Mostly, it may be because the actual model is not linear, and the predictors are not strictly independent to each other. 

In short, we may pick random forest as our final model. 
