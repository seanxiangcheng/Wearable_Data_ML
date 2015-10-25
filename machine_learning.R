#script.dir <- dirname(sys.frame(1)$ofile)
#setwd(script.dir)

setwd("/home/xcheng0907/GoogleDrive/Courses/DataScience_Coursera/PracticalMachineLearning/Wearable_Data_ML")
library(caret)

#read the data and find the variables to use
trainingRaw = read.csv("pml-training.csv")
NAsTrain = apply(trainingRaw, 2, function(x) {sum(is.na(x))})
col2keep = (NAsTrain<0.1*nrow(trainingRaw))
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
trainData = trainData[, colTrain2keep]
trainData$classe = trainingRaw$classe

# training data is splitted into training set and test set
inTrain = createDataPartition(trainData$classe, p=0.6, list=FALSE)
training = trainData[inTrain, ]
testing = trainData[-inTrain, ]

# Build model 
ctrl = trainControl(method="cv", number=5)
models = c("svmLinear", "rf", "gbm", "nb", "lda") 

modSVM = train(training$classe ~ ., data=training, method=models[1], trControl = ctrl)
predSVM = predict(modSVM, newdata=testing)
confusionMatrix(predSVM, testing$classe)


print(models[2])
modRF = train(classe ~ ., data=training, method=models[2], trControl=ctrl, prox=T, ntree=200)
predRF = predict(modRF, newdata=testing)
confusionMatrix(predRF, testing$classe)
print("Variables importance in model")
vi = as.data.frame(varImp(modRF$finalModel))
#vi = as.data.frame(vi[with(vi, order(vi$Overall, decreasing=TRUE)), ])


print(models[3])
modGBM = train(classe ~ ., data=training, method=models[3], trControl=ctrl, verbose=FALSE)
predGBM = predict(modGBM, newdata=testing)
confusionMatrix(predGBM, testing$classe)

print(models[4])
modNB = train(classe ~ ., data=training, method=models[4], trControl=ctrl)
predNB = predict(modNB, newdata=testing)
confusionMatrix(predNB, testing$classe)

print(models[5])
modLDA = train(classe ~ ., data=training, method=models[5], trControl=ctrl)
predLDA = predict(modLDA, newdata=testing)
confusionMatrix(predLDA, testing$classe)


### Test to the final test set
finalTestPred = list()
finalTestPred$SVM = predict(modSVM, newdata=finalTest)
finalTestPred$RF = predict(modRF, newdata=finalTest)
finalTestPred$GBM = predict(modGBM, newdata=finalTest)
finalTestPred$NB = predict(modNB, newdata=finalTest)
finalTestPred$LDA = predict(modLDA, newdata=finalTest)

# write predictions to file
finalTestPredDF = as.data.frame(finalTestPred)
write.table(finalTestPredDF, file="finalTestPred.csv", sep=",", row.names=FALSE)
