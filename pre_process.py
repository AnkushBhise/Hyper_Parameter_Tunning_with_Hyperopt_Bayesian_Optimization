from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


def div_into_x_y(data):
	# Use for Breast_Cancer_Wisconsin Data
	# x = data.iloc[:, 2:-1]
	# y = data.iloc[:, 1]
	
	# Use for BankNote_Authentication Data
	# x = data.iloc[:, :-1]
	# y = data.iloc[:, -1]
	
	# Use for Iris Data
	x = data.iloc[:, 1:-1]
	y = data.iloc[:, -1]
	
	return x, y


def label_encode_target(data):
	# Transforming Y
	# encode class values as integers
	encoder = LabelEncoder()
	encoder.fit(data)
	encoded_data = encoder.transform(data)
	# convert integers to dummy variables (i.e. one hot encoded)
	return np_utils.to_categorical(encoded_data)


def data_cleaning(data):
	# Write cleaning process inside this function
	return data


def data_pre_process(data):
	x, y = div_into_x_y(data)
	y = label_encode_target(y)
	return train_test_split(x, y, test_size=0.2, shuffle=True)
