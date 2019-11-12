import numpy as np
import pandas as pd
import statistics 
from statistics import mode 
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import scikitplot as skplt
import matplotlib.pyplot as plt

# setupData object used to keep track of test/training data and relating parameters to be used by classifiers
class setupData():

    def __init__(self):  
        self.data = None
        self.test = None
        self.train = None
        self.numClasses = 0
        self.num_classifiers = 0
        self.class_labels = dict()
        self.training_splits = dict()

    def main(self):
        # read in data and store in a dataframe
        self.data = pd.read_csv("hazelnuts.csv")

        # data normalised before splitting into test and training sets.
        data = self.normalise(self.data)

        #randomly split data into test and train sets
        self.random_split(data) 

        # get number of unique classes
        self.set_num_classes()

        # get number of classifiers needed for OnevsOne classification
        self.set_num_classifiers()  

        #split data for use by each classifier
        self.binary_split()

    # feature normalisation has no impact on logistic regression however it is necessary to avoid a buffer overflow error later on
    def normalise(self, data):
            copy = data.copy()
            # normalise the data in each column except for class column
            for feature in data.columns:
                if feature != "class":
                    copy[feature] = (data[feature] - data[feature].min()) / (data[feature].max() - data[feature].min())
                    # copy[feature] = (data[feature] - data[feature].mean()) / data[feature].std()
                else:
                    copy[feature] = data[feature]
            return copy 

    # method used to randomly split input data into test and training data in a 2/3 to 1/3 ratio 
    def random_split(self, data):
        self.train=data.sample(frac=0.667) 
        self.test=data.drop(self.train.index) 

    # method used to split and store training data into different groups. Each group contains training data for only two classes. 
    def binary_split(self):
        for i in range(self.numClasses):
            data = self.train[self.train['class'] != self.class_labels[i]]
            self.training_splits[i] = data
    
    # method used to calculate number of unique classes that are present in input data
    # creates a dictionary containing class labels and respective numerical key
    def set_num_classes(self):
        cl = list(set(self.train["class"])) 
        self.numClasses = len(cl) 
        result = dict()
        for i in range(self.numClasses):
            result[i] = cl[i]
        self.class_labels = result
    
    # method to calculate the number of classifiers needed for OnevsOne classification depending on number of unique classes
    def set_num_classifiers(self):
        N = int(self.numClasses)
        self.num_classifiers = int((N*(N - 1))/2)


class logistic_classifier():

    def __init__(self): 
        self.parameters_optimal = None 
        self.X = []
        self.transform_y = dict()

    def train(self,X, y,iterations,learning_rate):
        # data transformed. labels encoded and matrix manipulation is done
        X, y = self.transform_data(X,y)

        # array of zeros 
        parameters = np.zeros((np.size(X,1),1))

        # optimise parameters using gradient descent
        # set parameters_optimal attribute for use in predictions 
        self.parameters_optimal = gradient_descent(X, y, parameters, learning_rate, iterations)

    # method used to make predictions
    def predict(self,X): 
        X = np.hstack((np.ones((len(X),1)),X))
        # test data and optimal paramters are applied to the sigmoid function which gives value between 1 and 0
        # results from sigmoid is rounded and this is the predictions
        temp_predictions = np.round(sigmoid(X @ self.parameters_optimal))
        
        prediction = []
        #predictions are transformed back to string values before being returned
        for x in temp_predictions:
            prediction.append(self.reverse_transform(x))
        return prediction
    
    # method to decode datalabels from assigned 1 or 0 back to string value
    def reverse_transform(self, x):
        reverse_transform = dict((v,k) for k,v in self.transform_y.items())
        return reverse_transform[int(x)]

    # method to encode data labels and put into form needed for training
    def transform_data(self, X, y):
        self.transform_y = dict([(k,j) for j,k in enumerate(sorted(set(y)))])
        y = np.asarray([self.transform_y[x] for x in y])
        y = y[:,np.newaxis]
        m = len(y)
        self.X = np.hstack((np.ones((m,1)),X))
        return self.X , y

# sigmoid/logit function output value between 0 and 1
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# gradient descent is used to find global minimum during parameter optimisation
# parameters get udpated 'num_iterations' times before being returned
def gradient_descent(X, y, parameters, learning_rate, num_iterations):
    for i in range(num_iterations):
        # cost gradient function
        cost_gradient = (1/len(y) * (X.T @ (sigmoid(X @ parameters) - y)))
        # parameter rate of change is relative to the learning rate
        parameters = parameters - (learning_rate)*(cost_gradient) 
    return parameters

def scikit_learn(X, y):
    clf = LogisticRegression(solver='lbfgs',multi_class='multinomial').fit(X, y)
    return clf

if __name__ == "__main__":
    
    all_ref_predictions = []
    all_predictions = []
    all_labels = []

    # creates a testing.txt file that will be use to analyse results while training 
    t= open("testing.txt","w+")
    t.write("Testing Predicted vs Actual")

    # create results.txt file to record the accuracies for each run and the average of them all
    r= open("results.txt","w+")

    total_score = 0
    total_ref_score = 0

    # Repeat the process 10 times. Each time hazelnuts.txt will be divided randomly into different test/train splits
    for Runs in range(0,10):

        # create and call setupData object to take in and process data for use in classifier testing and training
        data = setupData()
        data.main()

        # train scikit learn model using same training data
        model = scikit_learn(data.train.drop('class',1).values,data.train['class'].values)

        # empty array that will hold classifiers 
        classifiers = []

        # set learning rate and number of times optimisation will be ran
        iterations = 1000
        learning_rate = 0.05

        # num of classifers that is created varies on how many classe there are
        for i in range(0,data.numClasses):
            l = logistic_classifier()

            # train classifier using training data values and respective class labels
            l.train(data.training_splits[i].drop('class' , 1).values, data.training_splits[i]['class'].values,iterations,learning_rate)
            # append trained classifer onto array for use later
            classifiers.append(l)

        # empty array to store all predictions for each classifier
        predictions = []
        # loop through each trained classifier
        for l in classifiers:
            # pass test data without class labels into each classfier and record predictions
            predictions.append(l.predict(data.test.drop('class' , 1).values))

        # empty array to store final predictions
        final_predictions = []
        # set the number of tests equal to the size of the test data array
        num_test = len(data.test)

        for i in range (0, num_test):
            # store all classifier predictions for a single test data instance in results array
            result = []
            for j in predictions:
                result.append(j[i])

            # calculate the mode prediction for each test data instance and store in final_predictions array 
            # 'Unsure' is appended if there is no mode
            try:
                final_predictions.append(mode(result))
            except: 
                final_predictions.append("Unsure")

        all_predictions = all_predictions + final_predictions
        

        # make predictions on test data using scikit model
        p = model.predict(data.test.drop('class',1).values)
        all_ref_predictions = all_ref_predictions + list(p)

        actual = data.test['class'].values
        all_labels = all_labels + list(actual)

        # calculate score for run by summing the numer of times the predicted equals the actual and dividing by the total
        score = float(sum(final_predictions == actual))/ float(len(actual))

        # write run score to results file
        r.write("\nRun " + str(Runs)+ "  Score: " + str(score))

        # get score for reference algorithm
        ref_score = model.score(data.test.drop('class' , 1).values,data.test['class'].values)
        r.write("   Reference Score: " + str(ref_score))

        # keep track of total score
        total_score = score + total_score
        total_ref_score = ref_score + total_ref_score

        # write run score to testing file
        t.write("\n\nRun " + str(Runs) + " Score: " + str(score) + "\nPredicted" + "          Actual")

        # loop through each test instance and write predicted vs actual values to testing file
        for i in range(0, num_test):
            t.write("\n" + str(final_predictions[i]) + "        " + str(actual[i]))

    # calculate and write average for 10 runs        
    average_score = total_score/10
    average_ref_score = total_ref_score/10
    r.write("\n--------------------------------   -----------------------------------")
    r.write("\nAverage Score: " + str(average_score) + "  Average Reference Score: " + str(average_ref_score))

    # confusion matrix for my algorithm
    skplt.metrics.plot_confusion_matrix(all_labels, all_predictions, normalize=False, title='My Implementations Confusion matrix')

    #confusion matrix for reference algorithm
    skplt.metrics.plot_confusion_matrix(all_labels, all_ref_predictions, normalize=False, title='Reference Confusion matrix')
    plt.show()