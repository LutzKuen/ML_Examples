# Pandas is used for data manipulation
import pandas as pd
# Use numpy to convert to arrays
import numpy as np
# Import matplotlib for plotting and use magic command for Jupyter Notebooks
import matplotlib.pyplot as plt
# Use datetime for creating date objects for plotting
import datetime
#import pydot

###### Read in data #######
features = pd.read_csv('temps.csv')

#### Anzeige von bestimmten Zeilen und Statistiken ######
# Anzeige der ersten 5 Zeilen inkl. überschriftenzeile
    #features.head(5)
# Descriptive statistics for each column
    #features.describe()
    #print('The shape of our features is:', features.shape)

####### Datenvorbereitung ###################
# One-hot encode the data using pandas get_dummies
# aus Mon,DI,Mit,.. werden Spalten mit 0 und 1
features = pd.get_dummies(features)
# Display the first 5 rows of the last 12 columns
    #features.iloc[:,5:].head(5)
# Labels are the values we want to predict = targets
labels = np.array(features['actual'])
# Remove the labels from the features
# axis 1 refers to the columns
features= features.drop('actual', axis = 1)
# Saving feature names for later use (for Feature Importance)
feature_list = list(features.columns)
# Convert to numpy array
features = np.array(features)
# Using Skicit-learn to split data into training and testing sets
from sklearn.model_selection import train_test_split
# Split the data into training and testing sets
train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 0.25, random_state = 42)

    #print('Training Features Shape:', train_features.shape)
    #print('Training Labels Shape:', train_labels.shape)
    #print('Testing Features Shape:', test_features.shape)
    #print('Testing Labels Shape:', test_labels.shape)

# Baseline zum Vergleich wie gut die Prediction im Vergleich zu den internen
# Vorhersagen ist
# The baseline predictions are the historical averages
baseline_preds = test_features[:, feature_list.index('average')]
# Baseline errors, and display average baseline error
baseline_errors = abs(baseline_preds - test_labels)
#print('Average baseline error: ', round(np.mean(baseline_errors), 2))

########## Modelimport #####################
# Import the model we are using
from sklearn.ensemble import RandomForestRegressor
# Instantiate model with 1000 decision trees and 42 ???
rf = RandomForestRegressor(n_estimators = 1000, random_state = 42)

######### Model trainieren ###################
# Train the model on training data
rf.fit(train_features, train_labels)
# Use the forest's predict method on the test data
predictions = rf.predict(test_features)


##### Model Validieren und Model accuracy bestimmen #################
# Calculate the absolute errors
errors = abs(predictions - test_labels)
# Print out the mean absolute error (mae)
    #print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')
# Calculate mean absolute percentage error (MAPE)
    #mape = 100 * (errors / test_labels)
# Calculate and display accuracy
    #accuracy = 100 - np.mean(mape)
    #print('Accuracy:', round(accuracy, 2), '%.')


##### feature importances der input variablen bestimmen ##########
# importance intialisieren
importances = list(rf.feature_importances_)
# List of tuples with variable and importance
feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]
#Sortiert the feature_importances by most important first
feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
# Print out the feature and importances
[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances]

####Schreiben in Datei #######
#(1) Variante
#outFile = open('feature_importance.txt', 'w')
# SChreibt gesamte Liste in eine Zeile
    #outFile.write("%s\n" % feature_importances)
#(2) Variante
# Schreibt alle Listeneinträge jeweils in eine Zeile
#for line in feature_importances:
    #write line in output file
 #   outFile.write(str(line))
  #  outFile.write("\n")
#(3) Variante
with open('feature_importance.txt', 'w') as outFile:
    # ListenTupel werden separiert und in eine Zeile geschrieben
    # jede Zeile wird mit \n verknüpft, was für einen Zeilenumbruch sorgt
    outFile.write('\n'.join('%s ; %s' % x for x in feature_importances))
outFile.close()



###Alternative Methode nachdem der 1. Random Forest durchgelaufen ist ######
# Idee ist, dass man nur noch die wichtigen Variablen nimmt und
# sich das Ergebnis anschaut
# New random forest with only the two most important variables
    #rf_most_important = RandomForestRegressor(n_estimators= 1000, random_state=42)
# Extract the two most important features
    #important_indices = [feature_list.index('temp_1'), feature_list.index('average')]
    #train_important = train_features[:, important_indices]
    #test_important = test_features[:, important_indices]
# Train the random forest
    #rf_most_important.fit(train_important, train_labels)
# Make predictions and determine the error
    #predictions = rf_most_important.predict(test_important)
    #errors = abs(predictions - test_labels)
# Display the performance metrics
    #print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')
    #mape = np.mean(100 * (errors / test_labels))
    #accuracy = 100 - mape
    #print('Accuracy:', round(accuracy, 2), '%.')



######## Visualisation   ######
# Import tools needed for visualization
from sklearn.tree import export_graphviz
# Pull out one tree from the forest
    #tree = rf.estimators_[5]
# Export the image to a dot file
#export_graphviz(tree, out_file = 'tree.dot', feature_names = feature_list, rounded = True, precision = 1)
# Use dot file to create a graph
#(graph, ) = pydot.graph_from_dot_file('tree.dot')
# Write graph to a png file
  #graph.write_png('tree.png')


###### Ergebnis plotten #############
####plot only one tree
# Limit depth of tree to 3 levels
    #rf_small = RandomForestRegressor(n_estimators=10, max_depth = 3)
    #rf_small.fit(train_features, train_labels)
# Extract the small tree
    #tree_small = rf_small.estimators_[5]
# Save the tree as a png image
    #export_graphviz(tree_small, out_file = 'small_tree.dot', feature_names = feature_list, rounded = True, precision = 1)
    #(graph, ) = pydot.graph_from_dot_file('small_tree.dot')
    #graph.write_png('small_tree.png');

#### Ergebnis plotten #####
# Set the style
    #plt.style.use('fivethirtyeight')
# list of x locations for plotting
    #x_values = list(range(len(importances)))
# Make a bar chart
    #plt.bar(x_values, importances, orientation = 'vertical')
# Tick labels for x axis
    #plt.xticks(x_values, feature_list, rotation='vertical')
# Axis labels and title
    #plt.ylabel('Importance'); plt.xlabel('Variable'); plt.title('Variable Importances');
#### Plotting with prediction
# Dates of training values
#months = features[:, feature_list.index('month')]
#days = features[:, feature_list.index('day')]
#years = features[:, feature_list.index('year')]
# List and then convert to datetime object
#dates = [str(int(year)) + '-' + str(int(month)) + '-' + str(int(day)) for year, month, day in zip(years, months, days)]
#dates = [datetime.datetime.strptime(date, '%Y-%m-%d') for date in dates]
# Dataframe with true values and dates
#true_data = pd.DataFrame(data = {'date': dates, 'actual': labels})
# Dates of predictions
#months = test_features[:, feature_list.index('month')]
#days = test_features[:, feature_list.index('day')]
#years = test_features[:, feature_list.index('year')]
# Column of dates
#test_dates = [str(int(year)) + '-' + str(int(month)) + '-' + str(int(day)) for year, month, day in zip(years, months, days)]
# Convert to datetime objects
#test_dates = [datetime.datetime.strptime(date, '%Y-%m-%d') for date in test_dates]
# Dataframe with predictions and dates
#predictions_data = pd.DataFrame(data = {'date': test_dates, 'prediction': predictions})
# Plot the actual values
#plt.plot(true_data['date'], true_data['actual'], 'b-', label = 'actual')

# Plot the predicted values
#plt.plot(predictions_data['date'], predictions_data['prediction'], 'ro', label = 'prediction')
#plt.xticks(rotation = '60');
#plt.legend()

# Graph labels
#plt.xlabel('Date'); plt.ylabel('Maximum Temperature (F)'); plt.title('Actual and Predicted Values');

###plot with errors for tendencies etc.
# Make the data accessible for plotting
#true_data['temp_1'] = features[:, feature_list.index('temp_1')]
#true_data['average'] = features[:, feature_list.index('average')]
#true_data['friend'] = features[:, feature_list.index('friend')]

# Plot all the data as lines
#plt.plot(true_data['date'], true_data['actual'], 'b-', label  = 'actual', alpha = 1.0)
#plt.plot(true_data['date'], true_data['temp_1'], 'y-', label  = 'temp_1', alpha = 1.0)
#plt.plot(true_data['date'], true_data['average'], 'k-', label = 'average', alpha = 0.8)
#plt.plot(true_data['date'], true_data['friend'], 'r-', label = 'friend', alpha = 0.3)

# Formatting plot
    #plt.legend(); plt.xticks(rotation = '60');

# Lables and title
    #plt.xlabel('Date'); plt.ylabel('Maximum Temperature (F)'); plt.title('Actual Max Temp and Variables');