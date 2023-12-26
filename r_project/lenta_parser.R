install.packages("rvest")

# Загрузка необходимых библиотек
library(rvest)

url <- "https://lenta.ru/news/2023/10/10"

start_date <- as.Date("2022-01-01")
end_date <- as.Date("2023-01-01")

date_range <- seq(start_date, end_date, by = "days")

dates <- sapply(date_range, function(x) format(x, "%Y/%m/%d"))



get_hrefs <- function(url) {
  webpage <- read_html(url)
  elements <- html_elements(webpage, "a.card-full-news._archive")
  hrefs <- sapply(elements, function(x) html_attr(x, "href"))

  return(hrefs)
}


get_content_news <- function(href) {
  news <- read_html(href)
  element <- html_element(news, "div.topic-body__content")
  content <- html_text(element)

  return(content)
}

get_title_news <- function(href) {
  news <- read_html(href)
  element <- html_element(news, "a.topic-header__item.topic-header__rubric")
  title <- html_text(element)

  return(title)
}



#contents <- data.frame(text = c())
contents <- list(c(), c())
for (date_string in dates) {
  print(date_string)
  url <- paste("https://lenta.ru/news/", date_string, sep = "")

  tryCatch(expr = {
    hrefs <- get_hrefs(url)
  },
  error = function(err) { 
    print("error to get hrefs")
  })

  for (href in hrefs) {
    url <- paste("https://lenta.ru", href, sep = "")
    len <- length(contents[[1]])

    tryCatch(expr = {
      content <- get_content_news(url)
      contents[[1]][len + 1] <- content
    },
    error = function(err) { 
      print("error to get content")
    })

    tryCatch(expr = {
      title <- get_title_news(url)
      contents[[2]][len + 1] <- title
    },
    error = function(err) { 
      print("error to get title")
    })

    #contents <- rbind(contents, text = content)

  }
}


data <- as.data.frame(contents)
colnames(data) <- c("text", "topic")
print(data)
write.csv(data, file = "news.csv", row.names = FALSE, sep = ",")