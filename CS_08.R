# Case-Study Title: Use Classification algorithms in financial markets (Stock Market Prediction)
# Data Analysis methodology: CRISP-DM
# Dataset: S&P-500 (The Standard and Poor's 500) Timeseries data from 2019 to 2022
# Case Goal: Create an automatic financial trading algorithm for S&P-500 index (Algorithmic Trading)


### Required Libraries ----
install.packages('ggplot2')
install.packages('quantmod')
install.packages('TTR')
install.packages('e1071')
library('ggplot2')
library('quantmod')
library('TTR')
library('e1071')


### Get financial market historical Data from Internet ----
quantmod::getSymbols(Symbols = '^GSPC', from = '2019-01-01', to = '2022-12-31', periodicity = 'daily', src = 'yahoo')  # load Historical data of S&P-500 from Yahoo Finance API
dim(GSPC)  # 1008 records, 6 variables


### Step 1: Business Understanding ----
 # know business process and issues
 # know the context of the problem
 # know the order of numbers in the business

#What we will do in this case:
 #first, we make a hypothesis about market
 #then, convert this hypothesis to an algorithm
 #then, measure this algorithm performance on Test data
 #then, use this algorithm for prediction (predict market direction -> 0: price will fall, 1: price will raise)
 #then, make a Decision-Rule based-on these predictions
 #finally, do trade via this Decision-Rule
 #now, we have a Trading Algorithm

#S&P-500 (Standard and Poor 500): 
 #an index of USA stock market
 #is a stock market index that measures the stock performance of 500 large companies listed on stock exchanges in the USA
 #it shows total direction of NASDAQ market
 #we want to Trade on changes of S&P-500


### Step 2: Data Understanding ----
### Step 2.1: Data Inspection (Data Understanding from Free Perspective) ----
## Dataset variables definition
colnames(GSPC)

#Index:	   the Timestamp of record (date)
#Open:	   the price of asset at market opening (start of day)
#High:	   the maximum price of asset in a day
#Low:	   the minimum price of asset in a day
#Close:	   the price of asset at market closing (end of day)
#Volume:   the trading volume of asset in a day
#Adjusted: 


### Step 2.2: Data Exploring (Data Understanding from Statistical Perspective) ----
## Overview of Dataframe
class(GSPC)
head(GSPC)
tail(GSPC)
dim(GSPC)
summary(GSPC)
View(GSPC)

## Data Visualization
chartSeries(GSPC, type = 'line', subset = '2022', theme = chartTheme('white'))  # S&P-500 price changes through time in 2022
chartSeries(GSPC, type = 'candlesticks', subset = '2020-6', theme = chartTheme('white'))

## Technical Indicators
# Simple Moving Average (SMA)
sma <- TTR::SMA(Cl(GSPC), n = 20)  # calculate SMA-20 for each day on Close price
head(sma)
tail(sma)

# Exponential Moving Average (EMA)
ema <- TTR::EMA(Cl(GSPC), n = 20)  # calculate EMA-20 for each day on Close price
head(ema)
tail(ema)

# Relative Strength Index (RSI)
rsi <- TTR::RSI(Cl(GSPC), n = 14)  # calculate RSI-14 for each day on Close price
head(rsi)
tail(rsi)

chartSeries(GSPC, type = 'candlesticks', subset = '2020-01::2020-06', theme = chartTheme('white'))
addSMA(n = 20, on = 1, col = 'red')
addEMA(n = 20, on = 1, col = 'blue')
addRSI(n = 14, maType = 'EMA')


### Step 3: Data PreProcessing ----
data0 <- data.frame(date = index(GSPC), coredata(GSPC))  # convert timeseries to dataframe
head(data0)

data <- data0[, c('date', 'GSPC.Close', 'GSPC.Volume')]  # our hypothesis about market is: maybe 'trading volume' is an indicator of future market direction
data$rsi <- rsi  # our hypothesis about market is: maybe 'RSI' is a good index which gives us signals about future market direction
colnames(data) <- c('date', 'close_price', 'volume', 'rsi')  # rename columns
head(data)
tail(data)

#Calculate daily asset return
data$d_return <- 0
for(i in 2:nrow(data)){  # first day has not any return (because we have not its previous day)
	data$d_return[i] <- data$close_price[i]/data$close_price[i-1] - 1
}

#Calculate volume change
data$volume_change <- 0
for(i in 2:nrow(data)){
	data$volume_change[i] <- data$volume[i]/data$volume[i-1] - 1
}

head(data)
tail(data)

#Plot S&P-500 daily return
ggplot(data = data, aes(x = date, y = d_return)) +
	geom_line() +
	xlab('Time') +
	scale_y_continuous(name = 'Daily Return of S&P-500')  # it is very noisy and seems that we can not predict it at all!

summary(data$d_return)
hist(data$d_return, breaks = 50)
mean(data$d_return)  # swings around 0
sd(data$d_return)

hist(data$volume_change, breaks = 50)  # plot daily volume change

View(data)  # we don't access today's information at the beginning of the day (except RSI), we access previous days information

#another hypothesis: maybe if we know daily returns for previous days, they give us a signal about today's return (direction of market)!
#another hypothesis: maybe if we know volume changes for previous days, they give us a signal about today's return (direction of market)!

#Add previous lags
#daily return lags (for 5 previous days)
data$r_lag1 <- 0
for(i in 2:nrow(data)){
	data$r_lag1[i] <- data$d_return[i-1]
}

data$r_lag2 <- 0
for(i in 3:nrow(data)){
	data$r_lag2[i] <- data$d_return[i-2]
}

data$r_lag3 <- 0
for(i in 4:nrow(data)){
	data$r_lag3[i] <- data$d_return[i-3]
}

data$r_lag4 <- 0
for(i in 5:nrow(data)){
	data$r_lag4[i] <- data$d_return[i-4]
}

data$r_lag5 <- 0
for(i in 6:nrow(data)){
	data$r_lag5[i] <- data$d_return[i-5]
}

head(data)

#volume change lags (for 5 previous days)
data$v_lag1 <- 0
for(i in 2:nrow(data)){
	data$v_lag1[i] <- data$volume_change[i-1]
}

data$v_lag2 <- 0
for(i in 3:nrow(data)){
	data$v_lag2[i] <- data$volume_change[i-2]
}

data$v_lag3 <- 0
for(i in 4:nrow(data)){
	data$v_lag3[i] <- data$volume_change[i-3]
}

data$v_lag4 <- 0
for(i in 5:nrow(data)){
	data$v_lag4[i] <- data$volume_change[i-4]
}

data$v_lag5 <- 0
for(i in 6:nrow(data)){
	data$v_lag5[i] <- data$volume_change[i-5]
}

head(data)

#add Market Trend (market direction)
data$trend <- ifelse(data$d_return > 0, 1, 0)
data$trend <- factor(data$trend)
head(data)

#remove first 14 rows (in-complete rows)
data <- data[-c(1:14),]
head(data,15)  # appropriate and complete data for ML model

# Divide Dataset into Train and Test 
train <- data[format(data$date, '%Y') %in% c('2019', '2020'),]  # for train models
head(train)
tail(train)

test <- data[format(data$date, '%Y') == '2021',]  # for test models
head(test)
tail(test)

real <- data[format(data$date, '%Y') == '2022',]  # for trade in real with models (simulate trading)
head(real)
tail(real)


### Step 4: Descriptive Analysis ----
## Correlation Analysis
#we can't calculate "pearson" correlation on trend (Binary-Categorical variable), but trend == d_return

cor_table1 <- round(cor(data[, c(5, 7:11, 4)]) , 2)  # pearson correlation between d_return & rsi & d_return lags
cor_table1  # rsi and r_lag1 have good linear relationship with d_return

cor_table2 <- round(cor(data[, c(5, 12:16)]), 2)  # pearson correlation between d_return & v_change lags
cor_table2


### Step 5: Modeling ----
# Model 1: Logistic Regression
model_rm1 <- glm(trend ~ rsi +
			r_lag1 + r_lag2 + r_lag3 + r_lag4 + r_lag5 + 
			v_lag1 + v_lag2 + v_lag3 + v_lag4 + v_lag5,
			family = 'binomial',
			data = train)
summary(model_rm1)
#consider coefficients based on Wald-test results: choose significant variables to be present on next model

model_rm2 <- glm(trend ~ rsi +
			r_lag1 + r_lag2 + r_lag3 + r_lag4 + r_lag5,
			family = 'binomial',
			data = train)
summary(model_rm2)
#r_lag2 is partially Significant, so we keep it in the model

#Prediction on train
train$probs <- predict(model_rm2, train, type = 'response')  # output of Logistic Regression is p(Y=1)
head(train)

train$pred_logreg <- ifelse(train$probs >= 0.42, 1, 0)  # Threshold is 0.42
head(train)

mean(train$pred_logreg == train$trend) * 100  
#model preformance on Train dataset is 68.63% (in 68% of days in years 2019-2020 we had true prediction of market direction)

#confusion matrix
table(actual = train$trend, prediction = train$pred_logreg)
sum(train$trend == 1)  # 285 days market was ascending
sum(train$trend == 0)  # 206 days market was descending

#Prediction on test
test$probs <- predict(model_rm2, test, type = 'response')
head(test)

test$pred_logreg <- ifelse(test$probs >= 0.5, 1, 0)  # Threshold is 0.5
head(train)

mean(test$pred_logreg == test$trend) * 100
#model performance on Test dataset is 68.25% (in 68% of days in year 2021, model had true prediction of market direction)

#Model evaluation based-on 4 index of Confusion Matrix
#confusion matrix
confm_logreg <- table(actual = test$trend, prediction = test$pred_logreg)
confm_logreg

#Accuracy = TP + TN / Total
#TP = 126, TN = 46
(126 + 46) / nrow(test)  # 68% of days had true prediction of market direction

#Precision = TP / TP + FP
#TP = 126, FP = 63
126 / (126 + 63)  # 66% of days which the model predicted market is bullish, the market was truly bullish

#Sensitivity = TP / TP + FN
#TP = 126, FN = 17
126 / (126 + 17)  # 88% of days which market was bullish, the model predicted market is bullish truly -> 88% of days had success in bullish market prediction ***

#Specificity = TN / TN + FP
#TN = 46, FP = 63
46 / (46 + 63)  # 42% of days which market was bearish, the model predicted market is bearish truly

#result: this model has good performance in `bullish market` prediction

# Model 2: Random Forest
set.seed(1234)
model_rf <- randomForest::randomForest(trend ~ rsi +
						r_lag1 + r_lag2 + r_lag3 + r_lag4 + r_lag5 +
						v_lag1 + v_lag2 + v_lag3 + v_lag4 + v_lag5,
						data = train, mtry = 4, ntree = 500)

#Prediction on test
test$pred_rf <- predict(model_rf, test)
head(test)

mean(test$pred_rf == test$trend) * 100  # performance of model on Test dataset is 68.25% (in 68% of days in year 2021, model had true prediction of market direction)

#confusion matrix
confm_rf <- table(actual = test$trend, prediction = test$pred_rf)
confm_rf

#result: 
 #Random Forest has better performance overally
 #Random Forest has better performance on bearish market (has more successful prediction)
 #Logistic Regression has better performance on bullish market (has more successful prediction)

# Model 3: Naive Bayes Classifier
model_nb <- e1071::naiveBayes(trend ~ rsi +
					r_lag1 + r_lag2 + r_lag3 + r_lag4 + r_lag5 +
					v_lag1 + v_lag2 + v_lag3 + v_lag4 + v_lag5,
					data = train)
model_nb

#Prediction on test
test$pred_nb <- predict(model_nb, test)
head(test)
mean(test$pred_nb == test$trend) * 100  # performance of model on Test dataset is 59.92% (in 60% of days in year 2021, model had true prediction of market direction)

#confusion matrix
confm_nb <- table(actual = test$trend, prediction = test$pred_nb)
confm_nb

#result: 
 #Random Forest has better performance overally
 #Random Forest has better performance on bearish market (has more successful prediction)
 #Naive Bayes has better performance on bullish market (has more successful prediction)

# Model 4: Linear Discriminant Analysis (LDA)
model_lda <- MASS::lda(trend ~ rsi + 
				r_lag1 + r_lag2 + r_lag3 + r_lag4 + r_lag5 +
				v_lag1 + v_lag2 + v_lag3 + v_lag4 + v_lag5,
				data = train)
model_lda

#Prediction on test
test$pred_lda <- predict(model_lda, test)$class
head(test)
mean(test$pred_lda == test$trend) * 100  # prediction performance of LDA on Test dataset is 67% (in 67% of days in year 2021, model had true prediction of market direction)

#confusion matrix
confm_lda <- table(actual = test$trend, prediction = test$pred_lda)
confm_lda

#result: 
 #Random Forest has better performance overally
 #Random Forest has better performance on bearish market (has more successful prediction)
 #Naive Bayes has better performance on bullish market (has more successful prediction)

# Model 5: Support Vector Machine (SVM)
set.seed(1234)
tune_out <- e1071::tune('svm', trend ~ rsi +
					r_lag1 + r_lag2 + r_lag3 + r_lag4 + r_lag5 +
					v_lag1 + v_lag2 + v_lag3 + v_lag4 + v_lag5,
					data = train, kernel = 'polynomial',
					ranges = list(degree = c(2, 3, 4, 5, 10)))
summary(tune_out)
model_svm <- tune_out$best.model  # select best model with Minimum Error

#Prediction on test
test$pred_svm <- predict(model_svm, test)
head(test)
mean(test$trend == test$pred_svm) * 100  # prediction performance of SVM on Test dataset is 55.95% (in 56% of days in year 2021, model had true prediction of market direction)

#confusion matrix
confm_svm <- table(actual = test$trend, prediction = test$pred_svm)
confm_svm

#result: 
 #Random Forest has better performance overally
 #Random Forest has better performance on bearish market (has more successful prediction)
 #SVM has better performance on bullish market (has more successful prediction)

# Voting System
#we have 5 learner (Domain Expert) and they vote for market direction in each day. we consider overall result of their votes as final market direction prediction in that day.
test$pred_votsys <- ifelse(apply(test[, 19:23], 1, function(x) sum(x == 1)) >= 3, 1, 0)  # prediction based-on simple voting system (gives equal weights to all of learners vote)
head(test)
mean(test$trend == test$pred_votsys) * 100  # prediction performance of VS on Test dataset is 67.85% (in 68% of days in year 2021, model had true prediction of market direction)

#confusion matrix
confm_votsys <- table(actual = test$trend, prediction = test$pred_votsys)
confm_votsys

### Step 6: Strategy Implementation
#now, assume that we are in '1 Jan 2022' and want to run our trading machine for actual prediction

#Summary of performances on test data:
 #Logistic Regression:           68.25
 #Random Forest:                 68.25
 #Naive Bayes Classifier:        59.92
 #Linear Discriminant Analysis:  67.06
 #Support Vector Machines:       55.95
 #Voting System                  67.05

#we use Random Forest for 2022

#Prediction on real
real$pred <- predict(model_rf, real)
head(real)
mean(real$trend == real$pred) * 100  # 65.33% -> model is robust

#Adding Rule to Algorithm
 #in each day:
  #if Random Forest predicts that tomorrow market is bullish, we will buy at opening of tomorrow (in Open Price) and sell at closing of it (in Close Price)
  #if Random Forest predicts that tomorrow market is bearish, we will sell at opening of tomorrow (in Open Price) and buy at closing of it (in Close Price)

head(data0)
real$open_price <- data0[match(real$date, data0$date), 'GSPC.Open']  # we need Open Price for each day because of our Rule
head(real)

#initial deposit: $1000 -> assume that you have $1000 in your broker account at `3 Jan 2022` and want to invest it -> can we increase it at end of the year 2022?
real$balance <- 0
real$balance[1] <- 1000  # we don't trade on first day, we start trading from second day
head(real)  # our balance at `3 Jan 2022` is $1000

#Simulate Market Trading: our trading machine trades every day based-on our Rule and prediction model!
for(i in 2:nrow(real)){
	if(real$pred[i] == 1){  # Rule 1 -> if we predict Y=1, market is bullish, buy at open-price and sell at close-price in each day
		real$balance[i] <- real$balance[i-1] * real$close_price[i] / real$open_price[i]
	}
	if(real$pred[i] == 0){  # Rule 2 -> if we predict Y=0, market is bearish, sell at open-price and buy at close-price in each day
		real$balance[i] <- real$balance[i-1] * real$open_price[i] / real$close_price[i]
	}
}

head(real)
tail(real)

#Balance changes over Time (balance vs. day)
plot(real$balance, type = 'line', main = 'Account Balance', ylim = c(900, 4000))
abline(h = 1000, col = 'red')  # balance at t0

real$balance[nrow(real)]/real$balance[1] * 100  # 349.74% increasing balance at end of year 2022










