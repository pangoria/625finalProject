library(jsonlite)
library(tidyverse)
library(tidytext)
library(textdata)
library(stringr)

#loading in data
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

d <- as.data.frame(d)

write_json(d, "/Users/jennabedrava/Desktop/sentiment_data.json")

#VISUALIZATIONS

#making word summary dataframe
word_summary <- words %>% group_by(word) %>%
  summarise(mean_overall = mean(overall), sent_score = max(value), count_word = n())

#word cloud with ggplot
ggplot(filter(word_summary), aes(x = mean_overall, y = sent_score)) +
  geom_text(aes(label = word, color = count_word, size=count_word),
            position= position_jitter()) +
  scale_color_gradient(low = "lightblue", high = "darkblue") +
  coord_cartesian(xlim=c(4,5)) + guides(size = FALSE, color=FALSE, scale = "none")

#word cloud
wordcloud(words = word_summary$word, freq = word_summary$count_word, scale=c(2, 0.5),
          max.words=300, colors=brewer.pal(8, "Dark2"))

#plotting distribution of sentiment scores
ggplot(d) + geom_histogram(aes(x=mean_score))
ggplot(d) + geom_bar(aes(x=overall))






