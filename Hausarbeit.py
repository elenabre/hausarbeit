#%% Import libraries
from cgi import test
from operator import index
import pickletools
from attr import define, attrs, attrib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import sqlalchemy as sa
from sqlalchemy import Column, Integer, Float, Date, String, VARCHAR, create_engine
from sqlalchemy.ext.declarative import declarative_base
import csv
from csv import reader
import math

#%% We have train and ideal data. To find the four best fits with least square 
# from ideal and train we first need to read the data and store it in a database.
def Load_Data(file_train):
    '''Load data from train.csv file and create dataframe'''
    data_train = csv.reader(file_train, delimiter=',')
    return data_train.tolist()
# create class train for dataframe train, set x as primary key
Base = declarative_base()

class read_data():
    '''class to create a new database.
    ...
    Attributes:
        name: name of the database
    '''
    def __init__(self, name):
        ''' Constructs the arrtibute for the new database.
        Parameters:
            name: name of the database'''
        self.name = name
class train(read_data):
    '''class to create train database'''
    pass
    x = Column(Integer, primary_key=True, nullable=True)
    '''x: primary key for the database'''

# load data from train.csv into sql database 'train.db' trough sqlalchemy engine
engine_train = create_engine('sqlite:///train.db')
Base.metadata.create_all(engine_train)
file_train = 'train.csv'
data_train = pd.read_csv(file_train)
data_train.to_sql(con=engine_train, name='train', if_exists='replace', index=False)
'''replace the database if it already exists, no index is needed'''

# load data from ideal.csv file and create dataframe
def Load_Data(file_ideal):
    data_ideal = csv.reader(file_ideal, delimiter=',')
    return data_ideal.tolist()

# create class ideal for dataframe ideal, set x as primary key
class ideal(read_data):
    pass
    x = Column(Integer, primary_key=True, nullable=True)

# load data from ideal.csv into sql database 'ideal.db' trough sqlalchemy engine
engine_ideal = create_engine('sqlite:///ideal.db')
Base.metadata.create_all(engine_ideal)
file_ideal = 'ideal.csv'
data_ideal = pd.read_csv(file_ideal)
data_ideal.to_sql(con=engine_ideal, name='ideal', if_exists='replace', index=False)


#%% create list of train y-columns
train_list = data_train.columns.tolist()
train_list.pop(0)
'''skip column x to only see results for y-columns'''

#%% create an empty dictionary to store the four ideal functions with the least square sum of the y-deviation with the train data.
dict_four_ideal = {'train_datasets': train_list, 'ideal_results': [], 'sqrt_y_dev': [], 'max_dev': [], 'max_dev_sqrt': []}

# create a for loop to find the four ideal functions with the least square sum of the y-deviation
for y in train_list:
    # create array of train data
    train_array = np.array(data_train[y])
    # create a list of the range from the 50 ideal functions, starting with column 1
    ideal_range = list(range(1, 51, 1))

    # create empty lists to store the data for the ideal functions
    ideal_results_list = []
    y_dev_array_sqrt_sum_list = []
    max_dev_list = []
    max_dev_sqrt_list = []
    # create a for loop to find the ideal functions and store the data in the lists
    for i in ideal_range:
        # define the ideal function for y{i} and store result in results-list
        ideal_results = f'y{i}'
        ideal_results_list.append(ideal_results)
        # create array of ideal data
        ideal_array = np.array(data_ideal[ideal_results])

        # calculate the y-deviation between the ideal function and the train data
        y_dev_array = ideal_array - train_array
        # calculate the sum of the square root of the y-deviation
        y_dev_array = np.absolute(y_dev_array)
        y_dev_array_sqrt = np.sqrt(y_dev_array)
        y_dev_array_sqrt_sum = np.sum(y_dev_array_sqrt)
        # store the square root of the sum of the y-deviation in the list
        y_dev_array_sqrt_sum_list.append(y_dev_array_sqrt_sum)
        # calculate the maximum of the y-deviation
        max_dev = np.amax(y_dev_array)
        # store the maximum of the y-deviation in the list
        max_dev_list.append(max_dev)
        # calculate the square root of the maximum of the y-deviation
        max_dev_sqrt = np.sqrt(max_dev)
        # store the square root of the maximum of the y-deviation in the list
        max_dev_sqrt_list.append(max_dev_sqrt)
    # find the least square of the ideal functions and store the data in the dictionary
    min_sqrt_dev = min(y_dev_array_sqrt_sum_list)
    dict_four_ideal['sqrt_y_dev'].append(min_sqrt_dev)
    # create an index to search for the ideal functions
    min_sqrt_dev_index = y_dev_array_sqrt_sum_list.index(min_sqrt_dev)
    # store the ideal functions in the dictionary
    dict_four_ideal['ideal_results'].append(ideal_results_list[min_sqrt_dev_index])
    # store the maximum of the y-deviation in the dictionary
    dict_four_ideal['max_dev'].append(max_dev_list[min_sqrt_dev_index])
    # store the square root of the maximum of the y-deviation in the dictionary
    dict_four_ideal['max_dev_sqrt'].append(max_dev_sqrt_list[min_sqrt_dev_index])

# create a dataframe from the dictionary
df_four_ideal = pd.DataFrame(dict_four_ideal)
print(df_four_ideal)

#%% get the values of data_ideal where df_four_ideal.ideal_results is the same as the column in data_ideal and store the values in a new dataframe
df_four_ideal_data = data_ideal[df_four_ideal.ideal_results]
# plot the ideal_results
x_ideal = data_ideal.x
ideal_1 = df_four_ideal_data.iloc[:,0]
'''y column of first ideal function'''
train_y1 = data_train.y1
ideal_2 = df_four_ideal_data.iloc[:,1]
'''y column of second ideal function'''
train_y2 = data_train.y2
ideal_3 = df_four_ideal_data.iloc[:,2]
'''y column of third ideal function'''
train_y3 = data_train.y3
ideal_4 = df_four_ideal_data.iloc[:,3]
'''y column of fourth ideal function'''
train_y4 = data_train.y4

#%%create four plots for the ideal functions
fig, axs = plt.subplots(2, 2, figsize=(10,10))
'''use subplots to create four plots'''
axs[0, 0].plot(x_ideal, ideal_1, 'r', label='Ideal Function')
axs[0, 0].plot(x_ideal, train_y1, 'b', label='Train Function')
axs[0, 0].set_title(df_four_ideal.ideal_results[0])
'''title of the plot shows the name of the ideal function'''
axs[0, 0].legend()
axs[0, 0].grid()
axs[0, 1].plot(x_ideal, ideal_2, 'r', label='Ideal Function')
axs[0, 1].plot(x_ideal, train_y2, 'b', label='Train Function')
axs[0, 1].set_title(df_four_ideal.ideal_results[1])
axs[0, 1].legend()
axs[0, 1].grid()
axs[1, 0].plot(x_ideal, ideal_3, 'r', label='Ideal Function')
axs[1, 0].plot(x_ideal, train_y3, 'b', label='Train Function')
axs[1, 0].set_title(df_four_ideal.ideal_results[2])
axs[1, 0].legend()
axs[1, 0].grid()
axs[1, 1].plot(x_ideal, ideal_4, 'r', label='Ideal Function')
axs[1, 1].plot(x_ideal, train_y4, 'b', label='Train Function')
axs[1, 1].set_title(df_four_ideal.ideal_results[3])
axs[1, 1].legend()
axs[1, 1].grid()
plt.show()

#%% analyze with test.csv. dict_four_ideal has four ideal functions. Every point from test.csv is compared to the ideal functions. 
# The ideal function with the least deviation is the ideal function for the point. 
# Every test point that has a higher deviation than the deviation of the ideal function with the train function by the factor of the square root of 2 gets dropped.
data_test = pd.read_csv('test.csv')
#Create dataframe with all ideal functions to compare it to test.csv
all_ideal_functions = pd.DataFrame().assign(x=x_ideal, y1=ideal_1, y2=ideal_2, y3=ideal_3, y4=ideal_4)
# create arrays for test data
test_x_array = np.array(data_test.x)
test_y_list = data_test.y.tolist()
test_y_array = np.array(test_y_list)
# create function to find the same or nearest values from test and ideal data
def find_nearest(array, value):
    '''compare array to a value'''
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    '''we look for the lowest difference between the array and the value - needs to be absolute value'''
    return array[idx]
# find same or nearest x and y value for test and ideal values through for loop and store in a list
nearest_x_list = []
nearest_x_y_list = []
for x in test_x_array:
    nearest_x = find_nearest(all_ideal_functions.x, x)
    '''use predefined function to find the nearest x value'''
    nearest_x_list.append(nearest_x)
    # create new dataframe with the nearest x values and ideal functions, create a list to find the nearest y value
    nearest_x_y = pd.DataFrame().assign(x=all_ideal_functions.x[all_ideal_functions.x == nearest_x], y1=all_ideal_functions.y1[all_ideal_functions.x == nearest_x], y2=all_ideal_functions.y2[all_ideal_functions.x == nearest_x], y3=all_ideal_functions.y3[all_ideal_functions.x == nearest_x], y4=all_ideal_functions.y4[all_ideal_functions.x == nearest_x])
    nearest_x_y_list.append(nearest_x_y)

# create arrays for each ideal function where the nearest x value was found 
nearest_x_y_array = np.array(nearest_x_y_list)
nearest_x_y1_array = np.array(nearest_x_y_list)[:,:,1]
'''second column of nearest_x_y_list is the first ideal y value'''
nearest_x_y1_array = nearest_x_y1_array.ravel()
'''ravel() flattens the array'''
nearest_x_y2_array = np.array(nearest_x_y_list)[:,:,2]
nearest_x_y2_array = nearest_x_y2_array.ravel()
nearest_x_y3_array = np.array(nearest_x_y_list)[:,:,3]
nearest_x_y3_array = nearest_x_y3_array.ravel()
nearest_x_y4_array = np.array(nearest_x_y_list)[:,:,4]
nearest_x_y4_array = nearest_x_y4_array.ravel()
# calculate the deviation of each ideal y and test y value and store as new array
dev_y_1_array = nearest_x_y1_array - test_y_array
dev_y_2_array = nearest_x_y2_array - test_y_array
dev_y_3_array = nearest_x_y3_array - test_y_array
dev_y_4_array = nearest_x_y4_array - test_y_array
# get the absolute value of the deviation
dev_y_1_array = np.absolute(dev_y_1_array)
dev_y_2_array = np.absolute(dev_y_2_array)
dev_y_3_array = np.absolute(dev_y_3_array)
dev_y_4_array = np.absolute(dev_y_4_array)
#create dataframe with all deviation data
dev_df = pd.DataFrame().assign(dev_y_1=dev_y_1_array, dev_y_2=dev_y_2_array, dev_y_3=dev_y_3_array, dev_y_4=dev_y_4_array)
# find min value column and min value column name and add it to dataframe
c = dev_df.columns
dev_df[c] = dev_df[c].apply(lambda x: pd.to_numeric(x))
'''convert to numeric'''
dev_df = dev_df.assign(min_dev = dev_df[c].min(axis=1), min_dev_name = dev_df[c].idxmin(axis=1))
'''find min deviation of all deviation in each row and the column name of this min deviation'''
# add x and y test values to dataframe
dev_df = dev_df.assign(x=test_x_array, y=test_y_array)
print(dev_df)

#%% compare deviation of four ideal with deviation of test data and store the values that do not
# exceed the four ideal daviation by the factor of sqrt(2) in a new dataframe
test_result = pd.DataFrame()
for i in dev_df.index:
    if dev_df.min_dev_name[i] == 'dev_y_1':
        '''if the min deviation is the first ideal deviation, calulate if the deviation is lower than the ideal deviation by the factor of sqrt(2)'''
        if dev_df.dev_y_1[i] <= df_four_ideal.max_dev[0]*math.sqrt(2):
            '''first column of max_dev is the max deviation of the first ideal function'''
            test_result = test_result.append(dev_df.loc[i])
            '''append the row with the lowest deviation to the new dataframe'''
    elif dev_df.min_dev_name[i] == 'dev_y_2':
        if dev_df.dev_y_2[i] <= df_four_ideal.max_dev[1]*math.sqrt(2):
            test_result = test_result.append(dev_df.loc[i])
    elif dev_df.min_dev_name[i] == 'dev_y_3':
        if dev_df.dev_y_3[i] <= df_four_ideal.max_dev[2]*math.sqrt(2):
            test_result = test_result.append(dev_df.loc[i])
    elif dev_df.min_dev_name[i] == 'dev_y_4':
        if dev_df.dev_y_4[i] <= df_four_ideal.max_dev[3]*math.sqrt(2):
            test_result = test_result.append(dev_df.loc[i])

#%% create dataframe with x, y, min_dev and min_dev_name values of test_result, where for each min_dev_name value the y of ideal is shown
test_result_table = pd.DataFrame().assign(x=test_result.x, y=test_result.y, min_dev=test_result.min_dev, min_dev_name=test_result.min_dev_name,)
'''use the min_dev and min_dev_name values of test_result to find the y value of the ideal function that has the lowest deviation and drop the other columns'''
Results = {"dev_y_1": df_four_ideal.ideal_results[0], "dev_y_2": df_four_ideal.ideal_results[1], "dev_y_3": df_four_ideal.ideal_results[2], "dev_y_4": df_four_ideal.ideal_results[3]}
'''use Results to get the cideal y function column instead of 1, 2, 3, 4'''
test_result_table = test_result_table.assign(ideal_y=test_result_table.min_dev_name.map(Results))
'''map the min_dev_name values to the ideal_y values'''
test_result_table.drop(columns=['min_dev_name'], inplace=True)
'''drop the min_dev_name column'''      
print(test_result_table)

#%% store the test_result_table in database through sqlalchemy
engine = create_engine('sqlite:///test.db')    
test_result_table.to_sql('test', engine, if_exists='replace', index=False)

#%% plot x and y values of test_result_table with each associated ideal_y function
fig, axs = plt.subplots(2, 2, figsize=(10,10))
'''create a figure with 2x2 subplots'''
legend_elements = [Line2D([0], [0], color='r', lw=2, label='Ideal Function'),
                   Line2D([0], [0], marker='o', color='w', label='Test Point',
                          markerfacecolor='b', markersize=10)]
'''create a legend element for the ideal function and test point'''
axs[0, 0].plot(x_ideal, ideal_1, 'r')
'''plot the ideal function on the first subplot'''
for i in test_result_table.index:
    if test_result_table.ideal_y[i] == df_four_ideal.ideal_results[0]:
        axs[0, 0].plot(test_result_table.x[i], test_result_table.y[i], 'bo')
        '''Plot all test points that are associated with the first ideal function on the first subplot'''
axs[0, 0].set_title('ideal_y = ' + str(df_four_ideal.ideal_results[0]))
'''set the title of the first subplot to the ideal_y column name of the first ideal function'''
axs[0, 0].legend(handles=legend_elements, loc='lower right')
'''add the legend to the first subplot'''
axs[0, 1].plot(x_ideal, ideal_2, 'r')
for i in test_result_table.index:
    if test_result_table.ideal_y[i] == df_four_ideal.ideal_results[1]:
        axs[0, 1].plot(test_result_table.x[i], test_result_table.y[i], 'bo')
axs[0, 1].set_title('ideal_y = ' + str(df_four_ideal.ideal_results[1]))
axs[0, 1].legend(handles=legend_elements, loc='lower right')
axs[1, 0].plot(x_ideal, ideal_3, 'r')
for i in test_result_table.index:
    if test_result_table.ideal_y[i] == df_four_ideal.ideal_results[2]:
        axs[1, 0].plot(test_result_table.x[i], test_result_table.y[i], 'bo')
axs[1, 0].set_title('ideal_y = ' + str(df_four_ideal.ideal_results[2]))
axs[1, 0].legend(handles=legend_elements, loc='lower right')
axs[1, 1].plot(x_ideal, ideal_4, 'r')
for i in test_result_table.index:
    if test_result_table.ideal_y[i] == df_four_ideal.ideal_results[3]:
        axs[1, 1].plot(test_result_table.x[i], test_result_table.y[i], 'bo')
axs[1, 1].set_title('ideal_y = ' + str(df_four_ideal.ideal_results[3]))
axs[1, 1].legend(handles=legend_elements, loc='lower right')

plt.show()


# %%
