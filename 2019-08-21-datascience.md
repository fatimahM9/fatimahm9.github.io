---
title: "Data Science Project : movies rating & recommendation"
date: 2019-08-21
header:
  image: "/images/post-img/movies rating & recommendation/movies.jpg"
---
# Movies ratings & recommendations

## Introduction
One of my favorite things to do is watching movies and analyzing the story from a critique point and digging deeper into the meaning of the story and because I am learning data science I wanted to see the movies from different angle which is more in the  quantitative side so I decided to do some analytic and answer some questions in my mind.

In this project, I wanted to find features affecting the ratings of any particular movie and build a model to predict the movie ratings. also, make a function to take users top-rated movie (favorite) and give a recommendation to other users with a similar taste in movies.

**Some Questions I want to answers are:**
    1- what is the Age Distribution of the movie rater?
    2- how many stars each User rated the movie “Toy Story”?
    3- what are the Top 10 movies by viewership rating?
    4- can I make a function to recommend a movie to a user without machine learning?
    5- what are the feature that affects the rating of the movie? also, predict the movie ratings.

So here is what I found after my analysis.

## 1- what is the Age Distribution of the movie rater?
After cleaning and preprocessing the data now I could drive some insight from it. The first thing I wanted to find is the age distribution of the user who rated the movies. As we can see from the **fig.1** the age distribution for the users who rated the movies are mostly people in their 20s and 30s. Younger users don’t rate movies a lot.

[image:6300E38A-4EED-4851-9180-49F6CD99B6C8-6242-0000DD3F32D17184/unknown.png]
*Fig.1*

## 2- how many stars each User rated the movie “Toy Story”?
I wanted to analyze the data from different aspects to get further knowledge about the data so I picked a movie with the highest number of rating **fig.2** and in this case, it is **Toy Story** to see the distribution of rating for each start in 5 stars scale. To see the distribution of rating for that highest rated movie in this dataset and has high rating will make us understand the behavior of users.From **fig.3** it appear that a lot of users rated the movie Toy Story 4 and 5  which is obvious because **Toy Story** is a great movie from my bias opinion and few people rated it 1 and 2 stars and that I can’t understand.

[image:FED0B5B7-4D46-4551-8FA2-F047A82C51E4-6242-0000E7A5A1B6DF05/24AC12EA-ACCF-4A4A-867D-0991856C67D8.png]
*Fig.2*

[image:D29AB4E6-B286-4010-8022-2A1A1703A137-6242-0000DE3BB21AC48A/unknown.png]
*Fig.3*


## 3- what are the Top 10 movies by viewership rating?
This table **fig.4** shows us the top 10 movies with highest number rating and the highest average rating.
[image:C9C8B7A4-073F-48F0-A99F-CCE0EEE8DEA7-6242-0000DE7AACE50A3F/7BEB8051-C5A2-4D25-9F18-594214190A39.png]
*Fig.4*

From the plot**fig.5** we can see it more clear that the top viewed movie is **American Beauty** with almost 3500 ratings.

[image:412AF392-815D-4A39-9D98-5798AFB66C6E-6242-0000DE7CC965E185/unknown.png]
*Fig.5*

## 4- can I make a function to recommend a movie to a user without machine learning?
To make this function, there is further analysis to be done. I analyzed the data from the movie aspect and now it from the user aspect. So I chose one user, let’s take an example user with id = 2696.

*investigation on the user with ID = 2696:*
From the data, it shows that the user with ID = 2696 is a male between the age 25-34 with Occupation in executive/managerial has rated 20 movies.

### let’s Find the ratings for all the movies reviewed by the user of user-id = 2696:
In **fig.6** it shows all movie rated by this user and it appears that **Lone Star (1996)** is the highest in the list so I will assume that it is his favorite.

[image:1DEDA044-5784-49FE-944D-7139C9C62510-6242-0000DE45C15CBC8B/unknown.png]
*Fig.6*

**Note:** I will use this user after I made my recommendation function to recommend other movies for him.

after my analysis and understanding what features will help me in making the recommendations function now I can modify the dataframe to show only what I want with the help of a pivot table. The pivot table has user-id as index, movies title as columns and ratings as values.

Here is how the movie recommendation function works.  From the new pivot dataframe that I made I took the movie and found the correlation between the user who rated the movie with a high rating and other users then takes the user with a similar rating and show the top 5 similar movies and show it to the user. The code for the function is below if you want to look at it and understand it more.

```python
def recommendation_with_corr(movie, df=movies_pivot):
    '''
    Arguments:
    movie: name of the movie as a string
    df: name of the dataframe that the movie is in

    - you only need to write the name of the movie as a string

    '''
    ratings = df[movie]
    similar= df.corrwith(ratings)
    corr = pd.DataFrame(similar,columns=['correlation'])
    corr.dropna(inplace=True)
    sorted_corr = corr.sort_values(['correlation'],ascending=False)[:5]
    return sorted_corr
```

So now I can test the function with the user I analyze before. The analysis showed that his top-rated movie is “ Lone Star “ so I will use it to find similar movies.

*recommendation for user with ID =2696 based on his favorite “ Lone Star “:*
```
# top 5 movies similar to Lone Star
recommendation_with_corr('Lone Star (1996)')
```

Here are 5 movies that are similar to “ Lone Star “. **fig.7**  in term of ratings.

[image:56E31E77-34B6-49A7-9D57-3BCAC3FB342C-6242-0000DEA458997C01/C91435BE-CCF3-420B-9278-CD761B5E96D4.png]
*Fig.7*

This function is not very accurate and it needs more features other than just correlation between two similar ratings. But it is a start for further work.

## 5- predict the movie ratings.
So for this part  will use Gender, Age, Occupation and Genres as our features to predict the ratings, but  before that, I did some analysis on the ratings **fig.8**  we can see that the number of ratings is lower than the average rating. So the model will not be accurate. We need to make a threshold for how many rating is considered valid so our recommendation model is more accurate.

[image:92180868-BE1B-4E55-AACB-AA13978C623D-6242-0000DE6988ADD292/unknown.png]
*Fig.8*

Also if we analyze the plot above we could see that the majority of the ratings are between 3 and 4 stars in the left plot. From the right plot, it appears that a lot of users did not rate the movies, some of the ratings came from a few people.

#### We need to check if people rate a movie if they like it a lot and if the majority of rating is for popular movies.

So the scatter plot in **fig.9** confirms that the more ratings the movie has,  the more popular it is.

[image:D99D6C03-7DEB-498A-8FBF-EF65EEA6B50E-6242-0000DE70C568BADB/unknown.png]
*Fig.9*

Now after we understand the rating we could do some feature engineering to make the threshold. Here is how it is done:

    * set criteria for the number of ratings so if the number of ratings is low we will not consider it.
    * make a new column [weighted average rating](https://www.quora.com/How-does-IMDbs-rating-system-work) based on IMDB ratings.

weighted rating (WR) = (v ÷ (v+m)) × R + (m ÷ (v+m)) × C where:
R = average for the movie (mean) = (Rating)
v = number of votes for the movie = (votes)
m = minimum votes required to be listed
C = the mean vote across the whole report

```python
def weighted_rating(r=ratings_df['Rating'],m_votes = 20, v= ratings_df['number_of_ratings']):
    if v.size >= m_votes:
        c= r.mean()
        wr = round(( v / (v + m_votes) * r + (m_votes / (v +             m_votes)) * c))
        return wr
```

The last part is using supervised machine learning model to make the predictions and from the features that we had, it predicted the weighted ratings for the movie really good.**score above 0.6**

## conclusion
In conclusion, analyzing movies from the Quantitative perspective can give us more insight into what could give a movie a high rating. Rating of a movie affected by different factors, and in this post I analyzed some of them. Also how to make a simple recommendation function from a few features. It might not be very accurate but it is a starting point for understanding how to make a better movie recommendation model.
