import os
import datetime
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import metrics
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt
import seaborn as sns

from modules import tcrpair_config, mldata

def compile_train_plot(model, trainset, testset, datasetname, **kwargs):
	# Compiling, training and testing of a model

	model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[
		metrics.TruePositives(name='tp'),
		metrics.FalsePositives(name='fp'),
		metrics.TrueNegatives(name='tn'),
		metrics.FalseNegatives(name='fn'),
		metrics.BinaryAccuracy(name='accuracy'),
		metrics.Precision(name='precision'),
		metrics.Recall(name='recall'),
		metrics.AUC(name='AUC'),
	])

	starttime = datetime.datetime.now()
	mn_timestamp = '{}_{}'.format(datasetname, starttime.strftime('%Y%m%d-%H%M%S'))

	history_fit = model.fit(trainset, epochs=kwargs['e'], validation_data=testset)

	# Save model
	model.save(os.path.join(tcrpair_config.models_path, mn_timestamp))

	# Model log
	with open(os.path.join(tcrpair_config.log_ml_path, '{}.txt'.format(mn_timestamp)), 'w') as fh:
		fh.write('TensorFlow version: {}\n'.format(tf.__version__))
		fh.write('Dataset used: {}\n'.format(datasetname))
		fh.write('Size of shuffle buffer (size of whole dataset): {}\n'.format(kwargs['sfl']))
		fh.write('Batch size: {}\n'.format(kwargs['bs']))
		fh.write('Epochs: {}\n'.format(kwargs['e']))
		fh.write('Droput rate: {}\n'.format(kwargs['do']))
		model.summary(print_fn=lambda x: fh.write('{}\n'.format(x)))

	# Plot metrics for each epoch
	metrics_list = [['accuracy'], ['AUC'], ['precision'], ['recall']]
	metrics_df = pd.DataFrame(
		{'metric' : [x for l in [x * kwargs['e'] * 2 for x in metrics_list] for x in l],
		 'set'    : [x for l in [x * kwargs['e'] for x in [['train'], ['val']] * 4] for x in l],
		 'epoch'  : [x for x in [x for x in range(1, kwargs['e'] + 1)] * 8],
		 'value'  : [
			 x for l in [
				 history_fit.history[x[0]] + history_fit.history['val_{}'.format(x[0])
				 ] for x in metrics_list] for x in l
		 ]})
	# Annotate max AUCs
	sns.set(style='whitegrid')
	g = sns.relplot(x='epoch', y='value', hue='set', col='metric', col_wrap=2, data=metrics_df, kind='line')
	ax2 = g.axes[1]
	def add_auc_lines(ax, dataset, colorindex):
		c = sns.color_palette()[colorindex]
		max_auc_row = metrics_df.iloc[
			metrics_df.loc[(metrics_df.metric == 'AUC') & (metrics_df.set  == dataset)]['value'].idxmax()
		]
		ax.axhline(max_auc_row['value'], ls='--', alpha=0.6, color=c)
		ax.axvline(max_auc_row['epoch'], ls='--', alpha=0.6, color=c)
		ax.text(max_auc_row['epoch'], 0.8, 'epoch {}'.format(max_auc_row['epoch']), rotation=90, color=c)
		ax.text(10, max_auc_row['value'], 'AUC {:.2f}'.format(max_auc_row['value']), color=c)
		### FUNCTION END ###
	add_auc_lines(ax2, 'train', 0)
	add_auc_lines(ax2, 'val', 1)
	plt.savefig(
			os.path.join(tcrpair_config.mp_plots_path, '{}_metrics.png'.format(mn_timestamp)), bbox_inches='tight'
	)
	plt.close()

	# Plot loss
	plt.plot(history_fit.history['loss'])
	plt.plot(history_fit.history['val_loss'])
	plt.ylabel('loss')
	plt.xlabel('epoch')
	plt.legend(['train', 'validation'], loc='upper right')
	plt.savefig(
		os.path.join(tcrpair_config.mp_plots_path, '{}_loss.png'.format(mn_timestamp)), bbox_inches='tight'
	)
	plt.close()

	x_test = list()
	Y_test = list()
	for data_batch, labels_batch in testset.take(-1):
		for data_single, labels_single in zip(data_batch.numpy(), labels_batch.numpy()):
			x_test.append(data_single)
			Y_test.append(labels_single)
	x_test = np.asarray(x_test)

	# Plot ROC & precision/recall curve
	Y_true = Y_test
	Y_pred = model.predict(x_test).ravel()
	fpr, tpr, _ = roc_curve(Y_true, Y_pred)
	sns.set(style='whitegrid')
	sns.lineplot(fpr, tpr, label='model AUC {:.2f}'.format(roc_auc_score(Y_true, Y_pred)))
	sns.lineplot([0, 1], [0, 1], color='black', linestyle='--', alpha=0.6, label='random AUC 0.5')
	plt.xlabel('false positive rate')
	plt.ylabel('true positive rate')
	plt.legend()
	plt.savefig(
		os.path.join(tcrpair_config.mp_plots_path, '{}_roc.png'.format(mn_timestamp)), bbox_inches='tight'
	)
	plt.close()
	p, r, _ = precision_recall_curve(Y_true, Y_pred)
	sns.lineplot(p, r, label='model average precision {:.2f}'.format(average_precision_score(Y_true, Y_pred)))
	plt.xlabel('recall')
	plt.ylabel('precision')
	plt.legend()
	plt.savefig(
		os.path.join(tcrpair_config.mp_plots_path, '{}_precision_recall.png'.format(mn_timestamp)), bbox_inches='tight'
	)
	plt.close()

	return mn_timestamp
	### FUNCTION END ###

def single_input_model(xdata,
					   ydata,
					   epochs: int,
					   batch_size: int,
					   input_size: tuple,
					   units: int,
					   dropout_rate: float,
					   dataset_name: str):
	"""
	:parameter xdata -> from mldata.generate_dataset
	:parameter ydata -> ftom mldata.generate_dataset
	:parameter epochs -> number of epochs
	:parameter batch_size -> batch size
	:parameter input_size -> input shape for masking layer
	:parameter units -> LSTM units
	:parameter dropout_rate -> dropout rate
	:parameter dataset_name -> name of the dataset used
	"""

	# Defining parameters
	sfl = len(ydata)
	act = 'sigmoid'

	X_train, X_test, y_train, y_test = mldata.split_data(xdata, ydata)

	trainset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(sfl).batch(batch_size)
	testset = tf.data.Dataset.from_tensor_slices((X_test, y_test)).shuffle(sfl).batch(batch_size)

	model = tf.keras.models.Sequential()
	model.add(tf.keras.layers.Masking(mask_value=tcrpair_config.mask_value, input_shape=input_size))
	model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units, return_sequences=True)))
	model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units)))
	model.add(tf.keras.layers.Dropout(dropout_rate))
	model.add(tf.keras.layers.Dense(1, activation=act))

	# Compile, train and evaluate model
	timestamp = compile_train_plot(model, trainset, testset, dataset_name,
								   sfl = sfl, bs = batch_size, e = epochs, do = dropout_rate)

### FUNCTION END ###

def load_model(modelname: str):
	return tf.keras.models.load_model(os.path.join(tcrpair_config.models_path, modelname))
### FUNCTION END ###
