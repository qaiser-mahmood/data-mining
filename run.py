import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB

pd.options.mode.chained_assignment = None
pd.set_option('display.max_columns', None)  # or 1000
pd.set_option('display.max_rows', None)  # or 1000
pd.set_option('display.max_colwidth', -1)  # or 199
pd.set_option('display.width', 1200)
np.set_printoptions(threshold=sys.maxsize)


def train_knn(x_t, y_t):
    x_train, x_test, y_train, y_test = train_test_split(x_t, y_t, test_size=0.3, random_state=123)

    cross_validation_score = list()
    neighbors = list(range(3, 100, 2))
    for k in neighbors:
        knn = KNeighborsClassifier(n_neighbors=k)
        scores = cross_val_score(knn, x_train, y_train, cv=5, scoring='accuracy')
        cross_validation_score.append(scores.mean())

    miss_classification_err = [1 - x for x in cross_validation_score]
    optimal_k = neighbors[miss_classification_err.index(min(miss_classification_err))]
    validation_accuracy = round((1-min(miss_classification_err)), 2)
    knn_tuned = KNeighborsClassifier(n_neighbors=optimal_k)
    knn_tuned = knn_tuned.fit(x_t, y_t)

    # PLOT MISCLASSIFICATION ERROR VS k
    # UNCOMMENT BELOW LINES TO SEE THE GRAPH
    # print("The optimal number of neighbors is {}".format(optimal_k))
    # plt.plot(neighbors, miss_classification_err)
    # plt.title("KNN Parameter K Tuning")
    # plt.xlabel("Number of Neighbors K")
    # plt.ylabel("Miss Classification Error")
    # plt.show()
    return knn_tuned, validation_accuracy, optimal_k


def train_dt(x_t, y_t):
    x_train, x_test, y_train, y_test = train_test_split(x_t, y_t, test_size=0.3, random_state=123)

    cross_validation_score = list()
    depths = list(range(1, 100))
    for depth in depths:
        dt = DecisionTreeClassifier(max_depth=depth, min_samples_leaf=18, random_state=123)
        scores = cross_val_score(dt, x_train, y_train, cv=5, scoring='accuracy')
        cross_validation_score.append(scores.mean())

    miss_classification_err = [1 - x for x in cross_validation_score]
    optimal_depth = depths[miss_classification_err.index(min(miss_classification_err))]
    validation_accuracy = round((1-min(miss_classification_err)), 2)
    dt_tuned = DecisionTreeClassifier(max_depth=optimal_depth, min_samples_leaf=18, random_state=123)
    dt_tuned = dt_tuned.fit(x_t, y_t)

    # PLOT MISCLASSIFICATION ERROR VS OPTIMAL_DEPTH
    # UNCOMMENT BELOW LINES TO SEE THE GRAPH
    # print("The optimal depth of tree is {}".format(optimal_depth))
    # plt.plot(depths, miss_classification_err)
    # plt.title("Decision Tree Max Depth Parameter Tuning")
    # plt.xlabel("Max Depth")
    # plt.ylabel("Miss Classification Error")
    # plt.show()
    return dt_tuned, validation_accuracy, optimal_depth


def train_nb(x_t, y_t):
    x_train, x_test, y_train, y_test = train_test_split(x_t, y_t, test_size=0.3, random_state=123)

    nb = GaussianNB()
    scores = cross_val_score(nb, x_train, y_train, cv=5, scoring='accuracy')
    validation_accuracy = round(scores.mean(), 2)
    nb = nb.fit(x_t, y_t)

    return nb, validation_accuracy


def normalize_df(ds):
    # normalize the data
    for col in ds.columns[1:]:
        min_value = ds[col].min()
        max_min = (ds[col].max() - min_value).astype('float64')
        ds.loc[:, col] = (ds[col] - min_value) / max_min
    return ds


def discretize_df(ds):
    # change nominal data to numeric
    category_columns = ds.select_dtypes(['object']).columns
    for col in category_columns:
        ds.loc[:, col] = ds[col].astype('category')
        ds.loc[:, col] = ds[col].cat.codes
    return ds


def duplicate_att(ds):
    list_of_duplicate_att = list()

    # Iterate over all the columns in data set
    for i in range(ds.shape[1]):
        col = ds.iloc[:, i]  # Select ith column

        # Iterate over all the columns in data set from (i+1)th index till end
        for j in range(i + 1, ds.shape[1]):
            col2 = ds.iloc[:, j]  # Select jth column
            if col.equals(col2):  # Check two columns are equal
                list_of_duplicate_att.append({'duplicate': ds.columns.values[j], 'duplicate_of': ds.columns.values[i]})

    return list_of_duplicate_att


def missing_att(ds):
    ds_missing = 1 - (ds.count()/len(ds))
    missing_indices = np.where(0 < ds_missing)
    missing_att_names = ds.columns[missing_indices[0]]  # attribute names that contain missing values
    missing_att_values = ds_missing[missing_indices[0]] * 100.0  # change to percentage
    return missing_att_names, missing_att_values


def unchanged_att(ds):
    unique_value_count = ds.nunique()
    unchanged_att_indices = np.where(unique_value_count == 1)
    unchanged_att_names = ds.columns[unchanged_att_indices[0]]
    return unchanged_att_names


def fill_missing_values(ds):
    msg_atts = missing_att(ds)[0]
    for i in range(len(msg_atts)):
        if ds[msg_atts[i]].dtypes == object:
            ds[msg_atts[i]] = ds[msg_atts[i]].fillna(ds[msg_atts[i]].mode().iloc[0])
        else:
            temp = ds[msg_atts[i]].fillna(0)
            missing_values_count = len(temp) - ds[msg_atts[i]].count()
            column_mean = temp.sum()/(len(temp) - missing_values_count)
            ds[msg_atts[i]] = ds[msg_atts[i]].fillna(column_mean)
    return ds


def print_report(trained_model, test_ds):
    x_valid = test_ds[test_ds.columns[1:]]  # validation set without class labels
    y_valid = test_ds[test_ds.columns[0]]  # validation set class labels
    pred = trained_model.predict(x_valid)  # predict validation set class labels
    model_confusion = pd.crosstab(y_valid, pred, rownames=['Ground Truth'], colnames=['Predicted'], margins=True, margins_name='Total') # confusion metrics of model
    print()
    print('Classification Report of model on Test Set:')
    print(metrics.classification_report(y_valid, pred))
    print('----------------------------------------------')
    print()
    print('Confusion Metrics of model on Test Set:')
    print(model_confusion)
    print('================================================')


df = pd.read_csv('data2019.student.csv')
test_set_id = df.loc[1000:, 'ID']  # save ID column to be used in final results

# UNCOMMENT BELOW LINE TO SEE THE INFORMATION ABOUT DATA SET
# print(df.describe())
# print('*******************************************************')

# USEFUL INFORMATION ABOUT DATA
print('Duplicate Attributes: ', duplicate_att(df))
print('*******************************************')
print()
print('Duplicate Instances: ', np.count_nonzero(df.iloc[:, 1:].duplicated()))
print('*******************************************')
print()
print('Unchanged Attributes: ', unchanged_att(df))
print('*******************************************')
print()
print('Missing Value Attributes: ', missing_att(df)[0])
print('*******************************************')
print()
print('Percentage of missing Values: ')
print(missing_att(df)[1])
print('*******************************************')

# UNWANTED ATTRIBUTES
# ID: no useful information for training
# att13: 935 out of 1000 values are missing
# att19: 937 out of 1000 values are missing
# att14: only one value. No change in value
# att17: only one value. No change in value
# att8: duplicate of att1
# att24 duplicate of att18
# duplicate instances = 100

df = df.drop(columns=['ID', 'att8', 'att24', 'att13', 'att19', 'att14', 'att17'])  # Remove unwanted. See above comments
df = df.drop_duplicates(subset=df.columns[1:])  # Remove duplicate instances

# PREPARE TEST SET - attention - DO NOT MIX UP TEST DATA BECAUSE ID's HAVE BEEN REMOVED. WE DON'T WANT TO CHANGE THE ORDER NOW
test_set = df.loc[1000:]  # Separate test set
test_set = discretize_df(test_set)  # change nominal data to numeric
test_set = normalize_df(test_set)  # bring all the attributes within same range
test_set = test_set.replace(np.nan, 0, regex=True)  # replace NaN values to 0
df = df.drop(test_set.index)  # Remove test set instances from original data frame
test_set = test_set.drop(columns='Class')

# UNCOMMENT BELOW LINES TO SAVE TEST SET
f = open('test_set.csv', 'w')
f.write(test_set.to_csv(index=False))  # save the final result to .csv file
f.close()

# PREPARE VALIDATION SET
class0 = df.loc[df['Class'] == 0.0].sample(n=50, random_state=123)  # 50 random samples from class 0
class1 = df.loc[df['Class'] == 1.0].sample(n=50, random_state=123)  # 50 random samples from class 1
validation_set = class0.append(class1)  # combine both classes
validation_set = fill_missing_values(validation_set)  # fill missing data
validation_set = discretize_df(validation_set)  # change nominal data to numeric
validation_set = normalize_df(validation_set)  # bring all the attributes within same range
validation_set = validation_set.sample(frac=1, random_state=123)  # mix randomly
df = df.drop(validation_set.index)  # remove validation set instances from original data set

# UNCOMMENT BELOW LINES TO SAVE VALIDATION SET
f = open('validation_set.csv', 'w')
f.write(validation_set.to_csv(index=False))  # save the final result to .csv file
f.close()

# PREPARE TRAINING DATA SET
df = fill_missing_values(df)  # fill missing data
df = discretize_df(df)  # change nominal data to numeric
df = normalize_df(df)  # bring all the attributes within same range

df = df.sample(frac=1, random_state=123)  # mix the training set thoroughly

# UNCOMMENT BELOW LINES TO SAVE TRAINING SET
f = open('training_set.csv', 'w')
f.write(df.to_csv(index=False))  # save the final result to .csv file
f.close()

column_indices = df.columns
training_class_labels = df['Class']
df = df.drop(columns='Class')

# BALANCE THE TRAINING DATA SET
smote = SMOTE(ratio='minority', random_state=123)  # create SMOT model
df, lbls = smote.fit_sample(df, training_class_labels)  # create synthetic samples
df = np.hstack((lbls.reshape(-1,1), df))  # combine balanced data set with balanced class labels. This is important before mixing the balanced data

df = pd.DataFrame(df)  # change type from numpy ndarray to pandas Dataframe
df.columns = column_indices  # add column names at the top of the balanced data set
df = df.sample(frac=1, random_state=123)  # mix up the balanced data set thoroughly

training_class_labels = df['Class']
df = df.drop(columns='Class')  # remove Class attribute from the balanced data set. This step should be performed AFTER randomly mixing the balanced data.

knn_trained, knn_estimated_accuracy, best_k = train_knn(df, training_class_labels)  # Train a KNN model
dt_trained, dt_estimated_accuracy, best_depth = train_dt(df, training_class_labels)  # Train a Decision Tree model
nb_trained, nb_estimated_accuracy = train_nb(df, training_class_labels)  # Train a Naive Bayes model

# UNCOMMENT BELOW LINES TO SEE VALIDATION SET CLASSIFICATION REPORT FOR EACH MODEL
print('KNN estimated accuracy: ', knn_estimated_accuracy, ' Optimal_k: ', best_k)
print_report(knn_trained, validation_set)

print('Decision Tree estimated accuracy: ', dt_estimated_accuracy, ' Optimal_depth: ', best_depth)
print_report(dt_trained, validation_set)

print('Naive Bayes estimated accuracy: ', nb_estimated_accuracy)
print_report(nb_trained, validation_set)

# PREDICTIONS OF ACTUAL TEST DATA
knn_pred_test_set = knn_trained.predict(test_set)  # predict using KNN model
nb_pred_test_set = nb_trained.predict(test_set)  # predict using Naive Bayes model
test_set_id = np.array(test_set_id, dtype='int64').reshape(-1, 1)  # reformat ID column into 1D vector
knn_pred_test_set = np.array(knn_pred_test_set, dtype='int64').reshape(-1, 1)  # reformat KNN predictions into 1D vector
nb_pred_test_set = np.array(nb_pred_test_set, dtype='int64').reshape(-1, 1)  # reformat Naive Bayes predictions into 1D vector
final_result = np.hstack((test_set_id, knn_pred_test_set, nb_pred_test_set))  # stack three vectors beside each other
final_result = pd.DataFrame(final_result)  # change to data frame for prettier display
final_result.columns = ['ID', 'KNN_PREDICTIONS', 'NAIVE_BAYES_PREDICTIONS']  # final result column headings
print(final_result)

f = open('predict.csv', 'w')
f.write(final_result.to_csv(index=False))  # save the final result to .csv file
f.close()
