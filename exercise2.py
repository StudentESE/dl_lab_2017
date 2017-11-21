import tensorflow as tf
import numpy as np
import pickle as cPickle
import os
import gzip
import matplotlib as plot
plot.use('Agg')
import matplotlib.pyplot as plot
import time
import matplotlib.patches as legendPatch

platform = 0 # 0 = Pool 1 = MBP

def mnist(datasets_dir='./data'):
	if not os.path.exists(datasets_dir):
		os.mkdir(datasets_dir)
	data_file = os.path.join(datasets_dir, 'mnist.pkl.gz')
	if not os.path.exists(data_file):
		print('... downloading MNIST from the web')
		try:
			import urllib
			urllib.urlretrieve('http://google.com')
		except AttributeError:
			import urllib.request as urllib
		url = 'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
		urllib.urlretrieve(url, data_file)

	print('... loading data')
	# Load the dataset
	f = gzip.open(data_file, 'rb')
	try:
		train_set, valid_set, test_set = cPickle.load(f, encoding="latin1")
	except TypeError:
		train_set, valid_set, test_set = cPickle.load(f)
	f.close()

	test_x, test_y = test_set
	test_x = test_x.astype('float32')
	test_x = test_x.astype('float32').reshape(test_x.shape[0], 1, 28, 28)
	test_y = test_y.astype('int32')
	valid_x, valid_y = valid_set
	valid_x = valid_x.astype('float32')
	valid_x = valid_x.astype('float32').reshape(valid_x.shape[0], 1, 28, 28)
	valid_y = valid_y.astype('int32')
	train_x, train_y = train_set
	train_x = train_x.astype('float32').reshape(train_x.shape[0], 1, 28, 28)
	train_y = train_y.astype('int32')
	rval = [(train_x, train_y), (valid_x, valid_y), (test_x, test_y)]
	print('... done loading data')
	return rval

def cnn(features, labels, mode, params):

	# Input Layer
	input_layer = tf.reshape(features["x"], [-1, 28, 28, 1])
	
	# Our CNN consists of two convolutional layers (16 3 × 3 filters and a stride of 1), 
	# each followed by ReLU activations and a max pooling layer. 
	
	# Convolutional Layer 1
	convolutional_layer_1 = tf.layers.conv2d(
		inputs=input_layer,
		filters=params["num_filters"],
		kernel_size=[3, 3],
		padding="same",
		activation=tf.nn.relu)

	# Pooling Layer #1
	polling_layer_1 = tf.layers.max_pooling2d(
		inputs=convolutional_layer_1, 
		pool_size=[2, 2], 
		strides=2)

	# Convolutional Layer #2 and Pooling Layer #2
	convolutional_layer_2 = tf.layers.conv2d(
		inputs=polling_layer_1,
		filters=params["num_filters"],
		kernel_size=[3, 3],
		padding="same",
		activation=tf.nn.relu)
	polling_layer_2 = tf.layers.max_pooling2d(
		inputs=convolutional_layer_2, 
		pool_size=[2, 2], strides=2)

	# Dense Layer
	polling_layer_2_flat = tf.reshape(
		polling_layer_2,
		[-1, 7 * 7 * params["num_filters"]])
	
	# After the convolution layers we add a fully connected layer with 128 units 
	dense_layer = tf.layers.dense(
		inputs=polling_layer_2_flat, 
		units=128, 
		activation=tf.nn.relu)

	# Logits Layer
	logits = tf.layers.dense(
		inputs=dense_layer, 
		units=10)

	predictions = {
		 "classes": tf.argmax(
			input=logits, 
			axis=1),
		# and a softmax layer to do the classification.
		"probabilities": tf.nn.softmax(
			logits, 
			name="softmax_tensor")
	}

	if mode == tf.estimator.ModeKeys.PREDICT:
		return tf.estimator.EstimatorSpec(
			mode=mode, 
			predictions=predictions)

	#Calculate Loss (for both TRAIN and EVAL modes)
	onehot_labels = tf.one_hot(
		indices=tf.cast(labels, tf.int32), 
		depth=10)

	loss = tf.losses.softmax_cross_entropy(
		onehot_labels=onehot_labels, 
		logits=logits)

	# We train the network by optimizing the cross-entropy loss with stochastic gradient descent
	if mode == tf.estimator.ModeKeys.TRAIN:
		optimizer = tf.train.GradientDescentOptimizer(
			learning_rate=params["learning_rate"])
		train_op = optimizer.minimize(
			loss=loss,
			global_step=tf.train.get_global_step())
		return tf.estimator.EstimatorSpec(
			mode=mode, 
			loss=loss, 
			train_op=train_op)

	# Add evaluation metrics (for EVAL mode)
	eval_metric_ops = {
		"accuracy": tf.metrics.accuracy(
			labels=labels, 
			predictions=predictions["classes"])}

	return tf.estimator.EstimatorSpec(
			mode=mode, 
			loss=loss, 
			eval_metric_ops=eval_metric_ops)

def compareLerningRates(X_train, y_train, X_valid, y_valid,max_epochs):
	# save the results (validation performance after each epoch) 
	# and plot all learning curves in the same figure
	
	
	colors = ['b', 'r', 'y', 'g']

	# config legend and axis
	l_rate_01 = legendPatch.Patch(color='b', label='Learning Rate 0.1')
	l_rate_001 = legendPatch.Patch(color='r', label='Learning Rate 0.01')
	l_rate_0001 = legendPatch.Patch(color='y', label='Learning Rate 0.001')
	l_rate_00001 = legendPatch.Patch(color='g', label='Learning Rate 0.0001')
	plot.figure()
	plot.suptitle('Validation Loss for Lerning Rates')
	plot.xlabel('Epochs of Training')
	plot.ylabel('Losses of Cycle')
	plot.legend(handles=[
		l_rate_01,
		l_rate_001,
		l_rate_0001,
		l_rate_00001],loc=3)

	# have a look on the effect of the learning rate 
	# on the network’s perfor- mance. 
	# Try the following values for the learning rate: 
	# {0.1, 0.01, 0.001, 0.0001}
	axis = np.arange(max_epochs)
	rates = [0.1, 0.01, 0.001, 0.0001]
	losses = np.zeros(max_epochs)
	#with tf.device("/device:GPU:0"):

	for i in range(len(rates)):
	# Create the Estimator
		mnist_cnn = tf.estimator.Estimator(
			model_fn=cnn,
			model_dir="/tmp/mnist_convnet_model_lr"+str(i),
			params ={"num_filters": 16, "learning_rate": rates[i]})

		for j in range(max_epochs):
			# Train the model
			
			training_input = tf.estimator.inputs.numpy_input_fn(
					x={"x": X_train},
					y=y_train,
					batch_size=64,
					num_epochs=1,
					shuffle=True)
			t = mnist_cnn.train(
					input_fn=training_input,
					steps=1)

			evaluation_input = tf.estimator.inputs.numpy_input_fn(
				x={"x": X_valid},
				y=y_valid,
				num_epochs=1,
				shuffle=False)
			losses[j] = mnist_cnn.evaluate(input_fn=evaluation_input)["loss"]
			print("Validation loss in Epoch #{}: {} (Learning Rate: {})" .format(j, losses[j], rates[i]))
		# plot all learning curves in the same figure
		plot.plot(axis, losses, c=colors[i])

	#plot.show()
	print("Saving Plot as ./compareLerningRates.png")
	plot.savefig('compareLerningRates.png')

	return [max_epochs, losses]
	pass

def gpuTraining(nums,num_epochs):
	# Train your neural network on a GPU with the following number 
	# of filters: {8, 16, 32, 64, 128, 256}
	with tf.device("/device:GPU:0"):

		nums = [8, 16, 32, 64, 128, 256]
		
		# measure the runtime (by using python’s build-in function time.time()) 
		times = np.zeros(len(nums))

		for i in range(len(nums)):
			# Create the Estimator
			mnist_cnn = tf.estimator.Estimator(
				model_fn=cnn,
				model_dir="/tmp/mnist_convnet_model_time_gpu" + str(i),
				params ={"num_filters": nums[i], "learning_rate": 0.1})
			start_gpu = time.time()
 
			for j in range(num_epochs):
				training_input = tf.estimator.inputs.numpy_input_fn(
					x={"x": X_train},
					y=y_train,
					batch_size=64,
					num_epochs=1,
					shuffle=True)
				mnist_cnn.train(
					input_fn=training_input,
					steps=1)
			end_gpu = time.time()
			diff = end_gpu - start_gpu
			times[i] = diff
			print("{:.2} ms with GPU for {} filters" .format(diff,nums[i]))
		return times
	pass

def cpuTraining(nums,num_epochs):
	#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
	with tf.device('/device:CPU:0'):
		nums = [8, 16, 32, 64]
		times = np.zeros(len(nums))# [0, 0, 0, 0]

		for i in range(len(nums)):
			# Create the Estimator
			mnist_cnn = tf.estimator.Estimator(
				model_fn=cnn,
				model_dir="/tmp/mnist_convnet_model_time_cpu" + str(i),
				params ={"num_filters": nums[i], "learning_rate": 0.1})
			start_cpu= time.time()
 
			for j in range(num_epochs):
				training_input = tf.estimator.inputs.numpy_input_fn(
					x={"x": X_train},
					y=y_train,
					batch_size=64,
					num_epochs=None,
					shuffle=True)
				mnist_cnn.train(
					input_fn=training_input,
					steps=1)

			end_cpu = time.time()
			diff = end_cpu -start_cpu
			times[i] = diff
			print("{:.2} ms with CPU for {} filters" .format(diff,nums[i]))
		return times
	pass

def compareGPU_CPU(nums,num_epochs):
	# At last, run the same experiments again on a CPU 
	# (with only the following number of filters: 
	# {8, 16, 32, 64})
	gpuTimes = 0;
	gpuTime = 0;
	cpuTimes = 0;
	cpuTime = 0;
	print("CPU Training")
	'''
	try:
		with tf.device('/CPU:0'):
		  a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
		  b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
		c = tf.matmul(a, b)
		# Creates a session with log_device_placement set to True.
		sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
		# Runs the op.
		sess.run(c)
	except:
		print('No CPU usable for tensorflow on this machine!')
		exit()
	'''
	start_cpu= time.time()
	cpuTimes = cpuTraining(nums,num_epochs)
	cpuTime = time.time() - start_cpu
	print('CPU Time: {}'.format(cpuTime))

	if(platform == 1):
		# Mac Book Pro
		np.save('mbpCPU.txt',cpuTimes)
	if(platform == 0):
		# PoolPC
		np.save('poolGPU.txt',gpuTimes)
		np.save('poolCPU.txt',cpuTimes)

		# checking if there is a GPU available
		'''
		try:
			with tf.device('/device:GPU:0'):
			  a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
			  b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
			c = tf.matmul(a, b)
			# Creates a session with log_device_placement set to True.
			sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
			# Runs the op.
			sess.run(c)
		'''
		print('GPU Training')
		start_gpu= time.time()
		gpuTimes = gpuTraining(nums,num_epochs)
		gpuTime = time.time() - start_gpu
		print('GPU Time: {}'.format(gpuTime))

		# factor increased speed
		factor = cpuTime/gpuTime
		print('Pool GPU is {} times faster than Mac Book Pros CPU'.format(round(factor,0)))
		'''
		except:
			print('No GPU usable on this machine!')
			#exit()
		'''

	if(platform == 1):
		# Mac Book Pro
		np.save('mbpCPU.txt',cpuTimes)
	if(platform == 0):
		# PoolPC
		np.save('poolGPU.txt',gpuTimes)
		np.save('poolCPU.txt',cpuTimes)

	# load saved times
	if(platform == 1):
		# loading GPU from Pool savings
		gpuTimes = np.load('poolGPU.txt.npy')
	cpuTimesMBP = []
	if(platform == 0):
		## loading CPU from Mac Book Pro
		cpuTimesMBP = np.load('mbpCPU.txt.npy')

	print("GPU Times:")
	print(gpuTimes)

	# scater plot
	scaterPlotFilterRuntime(cpuTimes,gpuTimes,cpuTimesMBP,nums)
	pass

def scaterPlotFilterRuntime(cpuTimes,gpuTimes,cpuTimesMBP,filters):
	# Generate a scatter plot with the number of parameters
	# on the x-axis and on the runtime on the y-axis. 
	# Each filter size should be one point in this scatter plot.
	'''
	plot.figure()
	plot.suptitle("Computation of {} Filters".format([8, 16, 32, 64]))
	plot.xlabel("number / filters")
	plot.ylabel("time / ms")
	#gpu = legendPatch.Patch(color='b', label='GPU times')
	cpu = legendPatch.Patch(color='r', label='CPU times')
	plot.legend(handles=[cpu])
	
	plot.scatter(filters, cpuTimes, c='r')
	print("Saving Scater Plot as ./scaterPlotFilterRuntimeCPU.png")
	plot.savefig('scaterPlotFilterRuntimeCPU.png')
	plot.figure()
	plot.suptitle("Computation of {} Filters".format(filters))
	plot.xlabel("number / filters")
	plot.ylabel("time / ms")
	gpu = legendPatch.Patch(color='b', label='GPU times')
	plot.legend(handles=[gpu])
	plot.scatter(filters, gpuTimes, c='b')
	#plot.show()
	print("Saving Scater Plot as ./scaterPlotFilterRuntimeGPU.png")
	plot.savefig('scaterPlotFilterRuntimeGPU.png')
	'''
	plot.figure()
	plot.suptitle("Computation of {} Filters".format(filters))
	plot.xlabel("number / filters")
	plot.ylabel("time / ms")
	plot.scatter(filters, gpuTimes, c='b')
	plot.scatter(filters, cpuTimes, c='r')
	plot.scatter(filters, cpuTimesMBP, c='g')

	gpu = legendPatch.Patch(color='b', label='GPU times (Pool PC)')
	cpu = legendPatch.Patch(color='r', label='CPU times (Pool PC)')
	cpuMBP = legendPatch.Patch(color='g', label='CPU times (MBP)')

	plot.legend(handles=[gpu, cpu, cpuMBP])
	print("Saving Scater Plot as ./scaterPlotFilterRuntime.png")
	plot.savefig('scaterPlotFilterRuntime.png')
	pass


# load
Dtrain, Dval, Dtest = mnist()
X_train, y_train = Dtrain
X_valid, y_valid = Dval
# Downsample training data to make it a bit faster for testing this code
#n_train_samples = 10000
n_train_samples = X_train.shape[0]
train_idxs = np.random.permutation(X_train.shape[0])[:n_train_samples]
X_train = X_train[train_idxs]
y_train = y_train[train_idxs]
print("Comparing Lerning Rates")

maxEpochsAndLosses = compareLerningRates(X_train, y_train, X_valid, y_valid,100)
#compareGPU_CPU([8, 16, 32, 64],10)
compareGPU_CPU([8, 16, 32, 64, 128, 256],100)

