import time
import os
import matplotlib.pyplot as plt
from pre_process import data_pre_process
from keras_model import evaluate_network
from hyperopt import hp, fmin, tpe, Trials
from functools import partial
from hyperopt.pyll.base import scope

import pandas as pd

import warnings

warnings.filterwarnings('ignore')


def plot_loss_vs_epoch_for_best(loss_vs_epoch_data, validation_loss_per_epoch, output):
	plt.title("Loss_Function_Vs_Epochs")
	plt.xlabel("Epochs")
	plt.ylabel("Loss_function")
	plt.plot(loss_vs_epoch_data, 'o-', color="g", label="Train_Loss_function")
	plt.plot(validation_loss_per_epoch, 'o-', color="y", label="Validation_Loss_function")
	plt.legend(loc="best")
	plt.show(block=False)
	plt.savefig(os.path.join(output, "loss_vs_epoch_for_best.jpg"))


def main(data, output):
	x, x_test, y, y_test = data_pre_process(data)
	param_space = {
		"activation": hp.choice("activation", ["relu", "LeakyReLU", "softmax", "tanh"]),
		"hidden_units": scope.int(hp.quniform("hidden_units", 10, 100, 1)),
		"hidden_layers": scope.int(hp.quniform("hidden_layers", 1, 2, 1)),
		"epochs": scope.int(hp.quniform("epochs", 20, 200, 1))
	}
	objective_function = partial(evaluate_network, x=x, y=y, x_test=x_test, y_test=y_test)
	trials = Trials()
	best = fmin(fn=objective_function, space=param_space, algo=tpe.suggest, max_evals=20, trials=trials)
	trail_results = trials.results
	trail_data = pd.DataFrame([di["trail_data"] for di in trail_results]).sort_values("Test_Accuracy", ascending=False)
	train_loss_per_epoch = pd.Series(trail_data["train_loss_per_epoch"][0])
	validation_loss_per_epoch = pd.Series(trail_data["validation_loss_per_epoch"][0])
	trail_data = trail_data.drop(columns=["train_loss_per_epoch", "validation_loss_per_epoch"])
	plot_loss_vs_epoch_for_best(train_loss_per_epoch, validation_loss_per_epoch, output)
	trail_data.to_csv(os.path.join(output, "Results_for_every_HyperParameter_combination.csv"), index=False)


if __name__ == '__main__':
	data_file = pd.read_csv("data/Iris.csv")
	output = "Iris_Results"
	main(data_file, output)
