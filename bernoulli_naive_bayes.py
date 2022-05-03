from ppmi_data_process import get_ppmi_data
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB

# get PPMI data
participant_status, all_features = get_ppmi_data()

# split data into train and test
all_features_train, all_features_test, participant_status_train, participant_status_test = train_test_split(all_features, participant_status, test_size=0.2, random_state=0)

# train and run model
bnb = BernoulliNB()
participant_status_pred = bnb.fit(all_features_train, participant_status_train).predict(all_features_test)

num_predictions = all_features_test.shape[0]
num_correct_predictions = (participant_status_test == participant_status_pred).sum()
print('Accuracy: %d/%d = %.2f%%' % (num_correct_predictions, num_predictions, num_correct_predictions / num_predictions * 100))