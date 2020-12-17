# octave_anomaly_recommender
Anomaly detection and recommender systems

Anomaly detection:
=============

1.  Gaussian
    1. mean
    2. variance
    3. epsilon - how to determine 
    4. probability distribution
2. Anomaly detection v supervised learning i.e. classification
3. What features to use?
    1. plot features and see shape i.e. gaussian
    2. its ok if its not a gaussian
        1. but there are ways to convert a chart to a gaussian
        2. take log(x) , log(x + c) exp(x, 0.5) etc
    3. We want p(x) to be high for normal x but low for abnormal x.
    4. choose features that take on very high or very low values in case of an anomaly
        1. a super example of creating new features
        2. example (CPU* CPU) / NETWORK traffic
4. The concept of multivariate gaussian
    1. the above can miss anomalies where x1 and x2 are not anatomies in their feature but an anomaly as a combination
    2. So rather than having p(x1), p(x2) etc, we need a new p(X) parameterized by mew and sigma (covariance matrix)
    3. when to use which model

5. Practical:
    1. detect failing servers on a network.
    2. you collected m = 307 examples of how they were behaving,
    3. vast majority of these examples are  Normal, but there might also be some examples of servers acting anomalously within this dataset.
    4. You will use a Gaussian model to detect anomalous examples in your dataset.
        1. code for multivariate gaussian
    5. We have 2 examples - one a 2D dataset and a much bigger dimension data set.
    6. you will implement an algorithm to select the threshold " using the F1 score on a cross validation set.
    7. For each cross validation example, we will compute p(xcv).
    8. The code has a nice way of calculating epsilon i.e. MAX(p(x)) - MIN(p(x)) / 1000 to get a step
        1. stepsize = (max(pval) - min(pval)) / 1000;
    9. The loop starting from MIN(p(x)) to MAX(p(x)) in steps of the above.

Recommender systems:
==================
1. When features are available the algorithm is like linear regression.
2. Collaborative filtering
    1.  what if features of the movie are not available. Can they be learnt.
    2. How we do know the features of a movie i.e. action content, romantic content etc
    3. what if we knew the users choices i.e. how much they like action movies etc. So we know the thetas rather than the x’s
    4. the idea is that everyone collaboratives to learn better features of the movies.
    5. Low rank matrices
3. Now that I have found features of movies, how do i find related movies ?
    1. find the norm between two movies
4. Mean normalization
    1. In the Y output, there could be severe, cases where a user has not rated a movie
    2. So we first apply mean normalization to each Y i.e. Y = Y - mean
    3. We store these means in a new vector ‘mew
    4. We subtract this from every ‘y’ i.e. create  a new Y
    5. We use this to learn Theta transpose X. I.E. learn theta’s and x’s
    6. And finally our pridicted value = Theta transpose X + mean
        1. This implies that if a user has not rated any movie, the prediction will be mean value
5. Practical:
    1. Movie ratings - 943 users, 1682 movies
    2. Y is a 1682 x 943 matrix, so is R
    3. The objective of collaborative  filtering is to predict movie ratings for the movies that users have not yet rated, that is, the entries with R(i; j) = 0. This will allow us to recommend the movies with the highest predicted ratings to the user.
    4. Other matrices include X and theta
