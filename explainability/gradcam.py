import tensorflow as tf
import cv2
import numpy as np

def gradcam(model, img_array, last_conv_layer_name):
	grad_model = tf.keras.models.Model(
		[model.inputs],
		[model.get_layer(last_conv_layer_name).output, model.output]
	)

	with tf.GradientTape() as tape:
		conv_outputs, predictions = grad_model(img_array)
		loss = predictions[:, np.argmax(predictions[0])]

	grads = tape.gradient(loss, conv_outputs)
	pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

	heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)
	heatmap = np.maximum(heatmap, 0)
	heatmap /= np.max(heatmap)

	return heatmap
