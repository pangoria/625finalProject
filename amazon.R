library(jsonlite)
library(tidyverse)
library(tidytext)
library(textdata)
library(stringr)
library(wordcloud)
library(colorspace)
library(scales)
library(VGAM)
library(pacman)
library(aod)
library(stargazer)
library(rpart)
library(rpart.plot)
library(yardstick)
library(e1071)
library(naivebayes)
library(caret)
library(cvms)

#loading in data
setwd("/home/bedrava/biostat625/625finalProject/")
d <- stream_in(file("AMAZON_FASHION.json"))

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

#removing influential point
which.max(d$score)
d <- d[-842154, ]

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

set.seed(1045314)
#plotting most common words in all reviews
wordcloud(words = word_summary$word, freq = word_summary$count_word, scale=c(5, 1),
          max.words=300, colors=brewer.pal(8, "Dark2"))

#plotting most common words in reviews with average 3, 4, 5 star reviews
wordcloud(words = filter(word_summary, mean_overall > 3)$word,
          freq = filter(word_summary, mean_overall > 3)$count_word, scale=c(2, 0.5),
          max.words=100,  colors=brewer.pal(9, "PuRd")[5:9])

#plotting most common words in reviews with average 1 and 2 start reviews
wordcloud(words = filter(word_summary, mean_overall < 3)$word,
          freq = filter(word_summary, mean_overall < 3)$count_word, scale=c(2, 0.5),
          max.words=100, random.order = FALSE, random.color = FALSE,
          colors=brewer.pal(9, "YlGnBu")[5:9])

#plotting distribution of sentiment scores
ggplot(d) + geom_histogram(aes(x=mean_score), binwidth = 1, fill = "white", color = "black") +
  scale_x_continuous(limits = c(-5, 5), breaks=seq(-5, 5, 1)) +
  scale_y_continuous(labels = comma) +
  labs(x = "Sentiment Scores", y = "Count", title = "Distribution of Sentiment Scores") +
  theme_minimal() + theme(axis.text=element_text(size=12), axis.title=element_text(size=14),
                          plot.title = element_text(size=14, face="bold", hjust=.5))

#plotting distribution of star rating
ggplot(data = d) + geom_bar(aes(x=overall, fill = as.factor(overall), color = as.factor(overall))) +
  scale_y_continuous(labels = comma) +
  scale_fill_manual(values = c("#7fcdbb", "#253494", "#fff7f3", "#df65b0", "#980043")) +
  scale_color_manual(values = c("#7fcdbb", "#253494", "#fff7f3", "#df65b0", "#980043")) +
  labs(x = "Number of Stars", y = "Count", title = "Distribution of Customer Star Rating") +
  theme_minimal() + theme(legend.position = "none", axis.text=element_text(size=12), axis.title=element_text(size=14),
                          plot.title = element_text(size=14, face="bold", hjust=.5))

#making boxplot of sentiment scores by overall rating
ggplot(data=d, aes(x=as.factor(overall), y=mean_score, fill = as.factor(overall))) + geom_boxplot() +
  scale_fill_manual(values = c("#7fcdbb", "#253494", "#fff7f3", "#df65b0", "#980043")) +
  labs(x = "Customer Star Rating", y = "Sentiment Score",
       title = "Sentiment Score by Customer Rating", fill = "Number of Stars") +
  theme_minimal() + theme(axis.text=element_text(size=12), axis.title=element_text(size=14),
                          plot.title = element_text(size=14, face="bold", hjust=.5),
                          axis.line = element_line(colour = "grey50"))

# =========== SPLIT TRAINING/TESTING ===========
# 60% of the sample size
smp_size <- floor(0.60 * nrow(d))

d$overall <- factor(d$overall, levels = c("1", "2", "3", "4", "5"), ordered = TRUE)

## set the seed to make your partition reproducible
set.seed(12252021)
train_ind <- sample(seq_len(nrow(d)), size = smp_size)

train <- d[train_ind, ]
test <- d[-train_ind, ]

#saveRDS(test, file = "/home/bedrava/biostat625/625finalProject/test.json")
#saveRDS(train, file = "/home/bedrava/biostat625/625finalProject/train.json")
#saveRDS(d, file = "/home/bedrava/biostat625/625finalProject/d.json")

# =========== REGRESSION ANALYSIS ===========

#MULTINOMIAL REGRESSION
#full model
pred_star <- vglm(overall ~ score + count_words, data=train,
                  family = multinomial(refLevel = 3))
summary(pred_star)

#creating pretty output of full model
fullmod_summary <- summaryvglm(pred_star)
stargazer(fullmod_summary@coef3, type="html", out = "fullmodel.html")

#finding predicted values of test data
test <- mutate(test, predicted_multi = apply((predict(pred_star, type = "res", newdata = test)), 1, which.max))
table(test$predicted_multi)

#PROPORTIONAL ODDS MODEL
library(MASS)
library(car)
pom <- polr(overall ~ score + count_words, data = train)
summary(pom)

#checking proportional odds assumption
poTest(pom) #Proportional odds assumption does not hold

#predicting values for test data
test <- mutate(test, predicted_pom = apply(predict(pom, newdata = test, type = "p"), 1, which.max))
table(test$predicted_pom)

#DECISION TREE
star_tree <- rpart(overall ~ score + count_words, data = train, method = 'class')
rpart.plot(star_tree)
test$predicted_star <- predict(star_tree, newdata = test, type = "class")

table(test$predicted_star) #decision tree not accurate

#trying naive Bayes stuff
naivebayes <- multinomial_naive_bayes(x = model.matrix(~score + count_words, data = train),
                                      y = train$overall)
summary(naivebayes)

#predicting training set values with naivebayes
test <- mutate(test,
               predicted_nb = predict(naivebayes, newdata = model.matrix(~score + count_words, data = test), type = "class"))
table(test$predicted_nb)

#CONFUSION MATRIX FOR NAIVE BAYES
cm_nb <- confusionMatrix(test$predicted_nb, test$overall, dnn = c("Prediction", "Reference"))
confusionMatrix(test$predicted_nb, test$overall, dnn = c("Prediction", "Reference"))$byClass

#visualizing confusion matrix
plt <- as.data.frame(cm_nb$table)
plt$Prediction <- factor(plt$Prediction, levels=rev(levels(plt$Prediction)))
plt$Total_Values <- ifelse(plt$Reference == 1, 42173,
                           ifelse(plt$Reference == 2, 25140,
                                  ifelse(plt$Reference == 3, 37668,
                                         ifelse(plt$Reference == 4, 57442, 177892))))
plt$Proportion <- plt$Freq/plt$Total_Values

ggplot(plt, aes(Reference, Prediction, fill= Freq/Total_Values)) +
  geom_tile() + geom_text(aes(label=Freq)) +
  scale_fill_gradient(low="white", high="#009194") +
  labs(x = "Actual Star Rating",y = "Predicted Star Rating", fill = "Proportion Classified",
       title = "Naive Bayes Confusion Matrix") +
  scale_x_discrete(labels=c("1 Star","2 Stars","3 Stars","4 Stars", "5 Stars"), position = "top") +
  scale_y_discrete(labels=c("5 Stars","4 Stars","3 Stars","2 Stars", "1 Star")) +
  theme_classic() +
  theme(axis.text=element_text(size=10), axis.title=element_text(size=12),
        plot.title = element_text(size=14, hjust=.5),
        axis.line = element_line(colour = "white"))

#CONFUSION MATRIX FOR MULTINOMIAL MODEL
cm_multi <- confusionMatrix(as.factor(test$predicted_multi), test$overall, dnn = c("Prediction", "Reference"))

#visualizing confusion matrix
plt2 <- as.data.frame(cm_multi$table)
plt2$Prediction <- factor(plt2$Prediction, levels=rev(levels(plt2$Prediction)))
plt2$Total_Values <- ifelse(plt2$Reference == 1, 42173,
                            ifelse(plt2$Reference == 2, 25140,
                                   ifelse(plt2$Reference == 3, 37668,
                                          ifelse(plt2$Reference == 4, 57442, 177892))))
plt2$Proportion <- plt2$Freq/plt2$Total_Values

ggplot(plt2, aes(Reference, Prediction, fill= Freq/Total_Values)) +
  geom_tile() + geom_text(aes(label=Freq)) +
  scale_fill_gradient(low="white", high="#009194") +
  labs(x = "Actual Star Rating",y = "Predicted Star Rating", fill = "Proportion Classified",
       title = "Multinomial Regression Confusion Matrix") +
  scale_x_discrete(labels=c("1 Star","2 Stars","3 Stars","4 Stars", "5 Stars"), position = "top") +
  scale_y_discrete(labels=c("5 Stars","4 Stars","3 Stars","2 Stars", "1 Star")) +
  theme_classic() +
  theme(axis.text=element_text(size=10), axis.title=element_text(size=12),
        plot.title = element_text(size=14, hjust=.5),
        axis.line = element_line(colour = "white"))





