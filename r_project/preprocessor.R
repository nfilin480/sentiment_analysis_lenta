# Загрузка необходимых библиотек
install.packages("SnowballC")
library(tm)
library(SnowballC)

data_file <- "E:\\IT_projects\\data_lab\\news.csv"

preprocess <- function(text) {
  # Преобразование в нижний регистр
  text_corpus <- tm_map(text_corpus, content_transformer(tolower))

  # Удаление стоп-слов
  text_corpus <- tm_map(text_corpus, removeWords, stopwords("russian"))

  # Удаление пунктуации
  text_corpus <- tm_map(text_corpus, removePunctuation)

  # Применение стемминга (например, с помощью русского стеммера из пакета SnowballC)
  text_corpus <- tm_map(text_corpus, stemDocument, language = "russian")

  return(text_corpus)
}

corpus <- read.csv(data_file)
normalized_text <- list(c())

for (text in corpus[["text"]]){
  # Создание корпуса
  text_corpus <- Corpus(VectorSource(text))
  len <- length(normalized_text[[1]])
  print(len)

  normalized_text[[1]][len + 1] <- as.list(preprocess(text_corpus))[[1]]
}


corpus$normalized_text <- normalized_text[[1]]
corpus <- as.data.frame(corpus)
write.csv(corpus, "data.csv", row.names = FALSE, sep = ",")