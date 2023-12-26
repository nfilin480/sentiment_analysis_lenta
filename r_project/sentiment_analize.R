install.packages("sentimentr")
install.packages("tm")
install.packages("e1071")
install.packages("caret")
install.packages("randomForest")
install.packages("glmnet")

library(glmnet)
library(randomForest)
library(sentimentr)
library(tm)
library(e1071)
library(caret)
# Загрузка и предобработка данных
data <- read.csv("E:\\IT_projects\\data_lab\\datasets\\data_sentiment_normal.csv")
#data <- head(data, 3000)

data$sentiment <- ifelse(data$sentiment == "positive", 1, 
                              ifelse(data$sentiment == "negative", 2, 
                                    ifelse(data$sentiment == "neutral", 0, data$sentiment)))

data$sentiment <- factor(data$sentiment)
print(data$sentiment)
# Создание матрицы термов-документов
dtm <- DocumentTermMatrix(data$normalized)
m <- as.matrix(dtm)

# Подготовка данных для обучения
y <- data$sentiment
x <- m



# Разделение данных на обучающую и тестовую выборки
set.seed(123)

train_index <- sample(seq_len(nrow(x)), 0.8  *  nrow(x))
#print(train_index)
train_x <- x[train_index, ]
train_y <- y[train_index]
test_x <- x[-train_index, ]
test_y <- y[-train_index]

#print(train_index)
print("Training...")
#cat("Прогресс:", progress, "\r")
control <- trainControl(method = "cv", number = 10)
#print(control)
model <- best.svm(train_y ~ train_x, data = data.frame(train_x), kernel = "linear", C = 1, trControl = control, alpha = 3)

print("Compute metrics...")

prediction <- predict(model, newdata = test_x, type = "class")
# Оценка точности модели
print(prediction)

accuracy <- sum(test_y == prediction[length(test_y)]) / length(test_y)
# Вывод результатов
cat("Точность модели:", accuracy, "\n")
#print(metrics)