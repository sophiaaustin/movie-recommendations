"""
Name: movie_recommendations.py
Date: 03/31/2020
Author: Sophia Austin, Hannah Hudson, Kevin McDonald
Description: This program implements colaborative filtering for a movie recommendation system
Group 6
"""
import re
import math
import csv
from scipy.stats import pearsonr

class BadInputError(Exception):
    """Exception Handler deals with bad inputs. 
    If the user enters invalid information the code will skip over it instead of crashing."""
    pass

class Movie_Recommendations:
    # Constructor

    def __init__(self, movie_filename, training_ratings_filename):
        """
        Initializes the Movie_Recommendations object from 
        the files containing movie names and training ratings.  
        The following instance variables should be initialized:
        self.movie_dict - A dictionary that maps a movie id to
               a movie objects (objects the class Movie)
        self.user_dict - A dictionary that maps user id's to a 
               a dictionary that maps a movie id to the rating
               that the user gave to the movie.    
        """
        #initialize dictionaries
        self.movie_dict = {}
        self.user_dict = {}

        #open file containing movies, get rid of quotes and commas
        movie_file = open(movie_filename, "r")
        movie_file.readline()
        csv_reader = csv.reader(movie_file, delimiter = ',', quotechar = '"')
        for line in csv_reader:
            self.movie_dict[int(line[0])] = Movie(int(line[0]), line[1])
        movie_file.close()
        
        #open file containing ratings, get rid of quotes and commas
        training_ratings_file = open(training_ratings_filename, "r")
        training_ratings_file.readline()
        csv_reader = csv.reader(training_ratings_file, delimiter = ',', quotechar = '"')
        for line in csv_reader:
            #check that number was entered
            if re.fullmatch("[0-9]+", line[0]):
                #if user id is already in dictionary, store in user_dictionary
                if int(line[0]) in self.user_dict:
                    user_dictionary = self.user_dict[int(line[0])]
                #if not in dictionary, find it and put it in user_dictionary
                else:
                    self.user_dict[int(line[0])] = {}
                    user_dictionary = self.user_dict[int(line[0])]
            user_dictionary[int(line[1])] = float(line[2]) #set key(user id) = to rating value
            movie_object = self.movie_dict[int(line[1])] #take user id from dict, set = to movie object
            movie_object.users.append(int(line[0])) #add user id from line above to list of users
        training_ratings_file.close()

    def predict_rating(self, user_id, movie_id):
        """
        Returns the predicted rating that user_id will give to the
        movie whose id is movie_id. 
        If user_id has already rated movie_id, return
        that rating.
        If either user_id or movie_id is not in the database,
        then BadInputError is raised.
        """


        #user or movie id does not exist, raise bad input error
        if user_id not in self.user_dict or movie_id not in self.movie_dict:
            raise BadInputError

        #check if the movie was already rated and get rating out
        elif user_id in self.movie_dict[int(movie_id)].users:
            actual_rating = self.user_dict[user_id][movie_id]
            return actual_rating

        #compute similarity between movies and weight them to calculate a predicted rating
        else:
            #loop through all movies that user has watched
            total_weighted_rating = 0.00
            total_sim = 0.00
            for movie in self.user_dict[user_id]:
                #get user's rating of movie out of the user dictionary
                this_user_dictionary = self.user_dict[user_id]
                rating = this_user_dictionary[movie]
                #get similarity of that movie with the target movie (movie_id)
                user_sim = self.movie_dict[movie].get_similarity(movie_id, self.movie_dict, self.user_dict) 
                #multiply similarity by rating = x
                weighted_rating = float(user_sim) * float(rating)
                #variable to add weighted ratings
                total_weighted_rating += weighted_rating
                #keep running sum of product of sum and user similarity
                total_sim += user_sim
            
            #if sum of sim = 0 , default predicted_rating to 2.5 
            if total_sim == 0:
                predicted_rating = 2.5
            
            #weighted sum: add all of the x together and divide by similarities of all movies that user has watched
            else:
                weighted_sum = float(total_weighted_rating) / float(total_sim)
                predicted_rating = weighted_sum
        return predicted_rating

    def predict_ratings(self, test_ratings_filename):
        """
        Returns a list of tuples, one tuple for each rating in the
        test ratings file.
        The tuple should contain
        (user id, movie title, predicted rating, actual rating)
        """
        
        list_of_tuples = [] #initialize the list of tuples to be returned
        test_ratings_file = open(test_ratings_filename, "r") #open test ratings file
        test_ratings_file.readline() #skip header line

        for line in test_ratings_file: #loop through all lines of the files, 
            #assigning values for user id, movie title, predicted rating, actual rating
            tokens = line.split(",")
            user_id = int(tokens[0])
            movie_id = int(tokens[1])
            movie_title = self.movie_dict[movie_id].title
            predicted_rating = self.predict_rating(user_id, movie_id)
            actual_rating = float(tokens[2])
            #append elements in tuple to list
            list_of_tuples.append((user_id, movie_title, predicted_rating, actual_rating))

        return list_of_tuples

    def correlation(self, predicted_ratings, actual_ratings):
        """
        Returns the correlation between the values in the list predicted_ratings
        and the list actual_ratings.  The lengths of predicted_ratings and
        actual_ratings must be the same.
        """
        return pearsonr(predicted_ratings, actual_ratings)[0]
        
class Movie: 
    """
    Represents a movie from the movie database.
    """
    def __init__(self, id, title):
        """ 
        Constructor.
        Initializes the following instances variables.  You
        must use exactly the same names for your instance 
        variables.  (For testing purposes.)
        id: the id of the movie
        title: the title of the movie
        users: list of the id's of the users who have
            rated this movie.  Initially, this is
            an empty list, but will be filled in
            as the training ratings file is read.
        similarities: a dictionary where the key is the
            id of another movie, and the value is the similarity
            between the "self" movie and the movie with that id.
            This dictionary is initially empty.  It is filled
            in "on demand", as the file containing test ratings
            is read, and ratings predictions are made.
        """
        #define variables
        self.id = id
        self.title = title
        self.users = []
        self.similarities = {}

    def __str__(self):
        """
        Returns string representation of the movie object.
        Handy for debugging.
        """
        
        return 'Movie'

    def __repr__(self):
        """
        Returns string representation of the movie object.
        """

        return "User ID:" + str(self.id) + "," + "Movie Title:" + str(self.title) + "List of Users:" + (self.users) + "Dictionary of "


    def get_similarity(self, other_movie_id, movie_dict, user_dict):
        """ 
        Returns the similarity between the movie that 
        called the method (self), and another movie whose
        id is other_movie_id.  (Uses movie_dict and user_dict)
        If the similarity has already been computed, return it.
        If not, compute the similarity (using the compute_similarity
        method), and store it in both
        the "self" movie object, and the other_movie_id movie object.
        Then return that computed similarity.
        If other_movie_id is not valid, raise BadInputError exception.
        """
        #if other_movie_id is not in the movie dict raise BadInput
        if other_movie_id not in movie_dict.keys():
            raise BadInputError
        #if similarity computed return value
        if other_movie_id in self.similarities:
            return self.similarities[other_movie_id]
        else:
            #else calculate sim and return value
            sim = self.compute_similarity(other_movie_id, movie_dict, user_dict)
            #assign sim to movie_id and other_movie_id objects
            movie_dict[self.id].similarities[other_movie_id] = sim
            movie_dict[other_movie_id].similarities[self.id] = sim
            return sim
        

    def compute_similarity(self, other_movie_id, movie_dict, user_dict):
        """ 
        Computes and returns the similarity between the movie that 
        called the method (self), and another movie whose
        id is other_movie_id.  (Uses movie_dict and user_dict)
        """
        #initialize local variables
        diff=0
        sum_of_rating_dif = 0
        counter_number_of_ratings = 0 
        #get users who watched the same movie
        other_movie = movie_dict[other_movie_id]
        #calculate the sum of ratings and increase counter for each movie shared by both users
        for user in self.users:
            if user in other_movie.users:
                sum_of_rating_dif += abs(float(user_dict[user][self.id]) - float(user_dict[user][other_movie_id])) #computing difference in rating and adding them to sum_of_ratings
                counter_number_of_ratings += 1 #counts the number of people that have watched a specific movie
        #if a movie has no ratings (no user rated that movie) then return a similarity of 0
        if counter_number_of_ratings == 0:
            return 0
        else:
            diff = float(sum_of_rating_dif) / float(counter_number_of_ratings) #computes the differnce rating that we then plug into the simiarlity formula 
            return 1 - float((diff/4.5)) #returns the similarity value
    

if __name__ == "__main__":
    # Create movie recommendations object.
    movie_recs = Movie_Recommendations("movies.csv", "training_ratings.csv")

    # Predict ratings for user/movie combinations
    rating_predictions = movie_recs.predict_ratings("test_ratings.csv")
    print("Rating predictions: ")
    for prediction in rating_predictions:
        print(prediction)
    predicted = [rating[2] for rating in rating_predictions]
    actual = [rating[3] for rating in rating_predictions]
    correlation = movie_recs.correlation(predicted, actual)
    print(f"Correlation: {correlation}")    