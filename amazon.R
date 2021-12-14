library(jsonlite)
library(tidyverse)
library(tidytext)
library(textdata)
library(stringr)

#loading in data
setwd("~/Downloads/")
d <- stream_in(file("~/Downloads/AMAZON_FASHION_5.json"))

##DATA CLEANING

#remove rows with NA in reviewText
d <- d %>% drop_na(reviewText)

#remove punctuation
d$reviewText <- str_replace_all(d$reviewText,  "[^[:alnum:]]", " ")

#all lowercase
d$reviewText <- tolower(d$reviewText)

#remove whitespace
d$reviewText <- str_squish(d$reviewText)

#filter by distinct review and reviewer ID
d <- d %>% distinct(reviewerID, reviewText, .keep_all=T)

#add unqiue ID for each review
d <- mutate(d, ID= 1:length(reviewText))

               
##SENTIMENT ANALYSIS

#extract words that match AFINN lexicon
words <- tibble(d) %>%
         unnest_tokens(word, reviewText, token="words") %>%
         inner_join(get_sentiments("afinn"))

#adds score and mean score to d
d <- d %>%
    left_join(words %>%
    group_by(ID) %>%
    summarise(score = sum(value), mean_score = mean(value), count_words=n())) %>%
    replace_na(list(score = 0, mean_score=0, count_words=0))





