from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.losses import categorical_crossentropy
from keras.layers import LeakyReLU
from sklearn.model_selection import KFold
from hyperopt import STATUS_OK
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
from sklearn.metrics import accuracy_score, f1_score


# define baseline model
def baseline_model(input_size, output_size, activation, hidden_units, hidden_layers):
	if activation == "LeakyReLU":
		activation = LeakyReLU()
	# create model
	model = Sequential()
	model.add(Dense(hidden_units, input_dim=input_size, kernel_initializer='uniform'))
	model.add(Activation(activation))
	for i in range(hidden_layers):
		model.add(Dense(units=hidden_units, kernel_initializer='uniform'))
		model.add(Activation(activation))
	model.add(Dense(output_size, activation='softmax'))
	# Compile model
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model


def evaluate_network(param, x, x_test, y, y_test):
	# monitor = EarlyStopping(monitor='loss', min_delta=0.1, patience=100, verbose=0, mode='auto',
	# 						restore_best_weights=True)
	model = baseline_model(input_size=x.shape[1], output_size=y.shape[1], hidden_units=param["hidden_units"],
						   activation=param["activation"], hidden_layers=param["hidden_layers"])
	
	# epoch_needed = []
	train_loss_per_epoch = []
	validation_loss_per_epoch = []
	train_loss = []
	train_accuracy = []
	test_loss = []
	test_accuracy = []
	test_f1_score = []
	kf = KFold(n_splits=5, shuffle=True)
	for tr_idx, val_idx in kf.split(x):
		x_tr = x.iloc[tr_idx]
		y_tr = y[tr_idx]
		x_val = x.iloc[val_idx]
		y_val = y[val_idx]
		history = model.fit(x_tr, y_tr, verbose=0, epochs=param["epochs"], validation_data=(x_val, y_val))
		train_loss.append(np.mean(history.history["loss"]))
		train_loss_per_epoch.append(history.history["loss"])
		validation_loss_per_epoch.append(history.history["val_loss"])
		train_accuracy.append(np.mean(history.history["accuracy"]))
		# epoch_needed.append(monitor.stopped_epoch)
		y_predict = model.predict(x_test)
		# loss, accuracy, f1_score, precision, recall = model.evaluate(x_test, y_test, verbose=0)
		loss = categorical_crossentropy(y_test, y_predict)
		test_loss.append(np.mean(loss))
		test_f1_score.append(f1_score(np.argmax(y_test, axis=1), np.argmax(y_predict, axis=1), average="micro"))
		test_accuracy.append(accuracy_score(np.argmax(y_test, axis=1), np.argmax(y_predict, axis=1)))
	train_loss_per_epoch = np.mean(np.array(train_loss_per_epoch), axis=0)
	validation_loss_per_epoch = np.mean(np.array(validation_loss_per_epoch), axis=0)
	train_loss = np.mean(train_loss)
	train_accuracy = np.mean(train_accuracy)
	test_loss = np.mean(test_loss)
	test_accuracy = np.mean(test_accuracy)
	test_f1_score = np.mean(test_f1_score)
	trail_data = {
		**param, **dict(Train_Accuracy=round(train_accuracy * 100, 2), Test_Accuracy=round(test_accuracy * 100, 2),
						Train_Loss=round(train_loss, 3), Test_Loss=round(test_loss, 3),
						Test_f1_Score=round(test_f1_score, 3), train_loss_per_epoch=train_loss_per_epoch,
						validation_loss_per_epoch=validation_loss_per_epoch)
	}
	# epoch_needed = np.mean(epoch_needed)
	return {
		"loss": -1 * test_accuracy, "trail_data": trail_data, "status": STATUS_OK
	}
