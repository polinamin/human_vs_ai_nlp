# Project 3: Web APIs & NLP

---

### Overview
This project summarizes learnings across three concepts: Classification Modeling, Natural Language Processing and Data Wrangling/Acquisition.

The following steps were followed, in order to complete this analysis:
1. Use [PRAW](https://praw.readthedocs.io/en/stable/) to collect and store question-response pairs from the following subreddits:
* [AskHistory](https://www.reddit.com/r/AskHistory/)
* [askscience](https://www.reddit.com/r/askscience/)
* [careerguidance](https://www.reddit.com/r/cscareerquestions)
* [cscareerquestions](https://www.reddit.com/r/careerguidance/)
<br>These were labeled as **Human Answers**. A total of 5523 pairs were collected using this method.


2. Use [OpenAI API](https://openai.com/blog/openai-api) to feed the extracted Reddit prompts to ChatGPT, and store responses as **AI Answers**. 
Data was collected using:
* The "text-babbage-001" method
* Temperature of 0.6
* Batches of 20 prompts at a time within limited sample ranges. The code was activated 25 times (with additional runs to ensure continuity across range bounds) in order to cycle through all available prompts.
* A lag time of 0.03 seconds between every 20-prompt batch within an activated code run
* Max_tokens of 300
<br>All responses were merged into a single column. 


3. Human Answers were classified as 0 and AI Answers were classified as 1, to stack into a single column.

4. The following text was removed from AI Answers:
* "I am only a machine"
* "As an AI language model"

5. Rows were removed if the following text appeared in the Human Answers column:
* \[deleted\]
* \[removed\]

6. Data was split using train-test-split, and run through the following models (with and without Lemmatization). A best-fit model was identified for the non-lemmatized data. 
* Bernoulli
* Multinomial
* Logistical

The models were run first with a CountVectorizer and then a TF-IDF string pre-processing.
The models were run through a gridsearch pipeline, with the following parameters:
* max_features: \[5000\],
* min_df: \[2,5\],
* max_df: \[0.9, 0.95\],
* ngram_range' : \[(1,1), (1, 2), (2,2)\],
* stop_words: ['english']

Additional models (and selected parameters) that were run include:
* DecisionTree
* Random Forest
    * n_estimators: [100, 150, 200],
    * max_depth: [None, 3, 4, 5, 6, 7]

<br>Finally, ADA boosting was applied (with n_estimators: [100,1000]) in order to optimize the a potential model.

<br>Accuracy, specificity and sensitivity were compared across the best-fit model for non-lemmatized and lemmatized data, in order to understand how it impacts model results.


### Modeling

| **Lemmatization** | **Vectorizer** | **Model** | **Train** | **Test** | **Best** |
| --- | --- | --- | --- | --- | --- |
| No | CountVectorizer/ TFIDF | Bernoulli | 0.79702 | 0.78653 | 0.78474 |
| Yes | CountVectorizer/ TFIDF | Bernoulli  | 0.95474 | 0.96206 | 0.95226 |
| No | CountVectorizer/ TFIDF | Multinomial | 0.82988<br>0.90303 | 0.78319<br>0.85236 | 0.78016<br>0.83236 |
| Yes | CountVectorizer/ TFIDF | Multinomial  | 0.99652 | 0.99739 | 0.99615<br>0.83236 |
| No | CountVectorizer/ TFIDF | Logistic | 0.94804<br>0.95771 | 0.88471<br>0.89215 | 0.87737<br>0.87873 |
| Yes | CountVectorizer/ TFIDF | Logistic  | 0.95474 | 0.96206 | 0.95226|
| No | CountVectorizer/ TFIDF | DecisionTree | 0.99975<br>0.99975| 0.83004<br>0.82595 | -- |
| Yes | CountVectorizer/ TFIDF | DecisionTree  | 0.999752<br> -- | 0.99256<br> -- | -- |
| No | CountVectorizer/ TFIDF | Random Forest | 0.999752<br> -- | 0.90368<br>0.91000 | 0.90018 |
| Yes | CountVectorizer/ TFIDF | Random Forest  | 0.99975 | 0.99479 | 0.99504 |

**With ADA Boosting**

| **Lemmatization** | **Vectorizer** | **Model** | **Train** | **Test** | **Best** |
| --- | --- | --- | --- | --- | --- |
| No | CountVectorizer/ TFIDF | DecisionTree | 0.98648<br>0.99938 | 0.89103<br>0.89773 | -- |
| Yes | CountVectorizer/ TFIDF | DecisionTree  | 0.99975<br>0.99975 | 0.99516<br>0.99516 | -- |
| No | CountVectorizer/ TFIDF | Logistic | 0.97247<br>0.87637 | 0.90256<br>0.86091 | -- |
| Yes | CountVectorizer/ TFIDF | Logistic  | 0.99975<br>0.99764 | 0.99590<br>0.99665 | -- |


### Model Selection and Evaluation
The model that was selected as optimal for predicting whether a response to a Reddit question will be by a human or ChatGPT was:
- ADA Boosted Logistical Regression.

This model was selected due to its high scores for train and test data sets, as well as the proximity of the results between the two groups (i.e. minimized overfitting by adjusting for bias and variance). 

The model (with no lemmatization) attained an accuracy score of: 0.90256
<br>The model (with lemmatization) attained an accuracy score of: 1.0

The model (with no lemmatization) attained an specificity score of: 0.88105
<br>The model (with lemmatization) attained an specificity score of: 0.99171

The model (with no lemmatization) attained an sensitivity score of: 0.92464
<br>The model (with lemmatization) attained an sensitivity score of: 0.99590

Based on this high-level side-by-side comparison, it is evident that Lemmatizing data prior to pre-processing and modeling is moving all model evaluation criteria up, allowing us to attain a near perfect prediction model.

