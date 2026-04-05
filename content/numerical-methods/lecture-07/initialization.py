# Weight Initialization: Xavier and He
#
# With bad initialization, activations explode or vanish as they
# pass through layers, making training unstable or impossible.
#
# Variance propagation through a linear layer y = Wx:
#   Var(y_j) = n_in * Var(W) * Var(x)
#
# To keep Var(y) = Var(x), we need:  Var(W) = 1 / n_in
#
# Xavier/Glorot (tanh, sigmoid):  Var(W) = 2 / (n_in + n_out)
# He/Kaiming (ReLU):              Var(W) = 2 / n_in

import time
import numpy as np


def propagate_through_layers(n_layers, layer_sizes, init_variance,
							 activation='relu'):
	"""
	Simulate forward pass through a deep network and track
	the standard deviation of activations at each layer.

	Parameters
	----------
	n_layers : int
	    Number of layers.
	layer_sizes : list of int
	    Width of each layer (same size for simplicity).
	init_variance : float
	    Variance of weight initialization: W ~ N(0, init_variance).
	activation : str
	    'relu', 'tanh', or 'linear'.
	"""
	n = layer_sizes[0]
	# Start with unit-variance input
	x = np.random.randn(1000, n)
	stds = [np.std(x)]

	for layer in range(n_layers):
		n_in = layer_sizes[min(layer, len(layer_sizes) - 1)]
		n_out = layer_sizes[min(layer + 1, len(layer_sizes) - 1)]

		W = np.random.randn(n_in, n_out) * np.sqrt(init_variance)
		x = x @ W

		if activation == 'relu':
			x = np.maximum(0, x)
		elif activation == 'tanh':
			x = np.tanh(x)

		stds.append(np.std(x))

		# Early stop if values explode or vanish
		if np.std(x) > 1e10 or np.std(x) < 1e-10 or np.isnan(np.std(x)):
			# Fill remaining with the last value
			stds.extend([stds[-1]] * (n_layers - layer - 1))
			break

	return stds


def initialization_experiment():
	"""
	Compare different weight initialization strategies and their
	effect on activation variance through a deep network.
	"""
	print(f"\n{'='*70}")
	print(f"Weight Initialization: Variance Propagation Through Layers")
	print(f"{'='*70}")

	n_layers = 15
	width = 256
	layer_sizes = [width] * (n_layers + 1)

	# Different initialization strategies
	strategies = [
		('Too large (Var=1.0)',        1.0,             'relu'),
		('Too small (Var=0.01)',       0.01,            'relu'),
		('He/Kaiming (Var=2/n_in)',    2.0 / width,     'relu'),
		('Xavier (Var=2/(n_in+n_out))', 2.0 / (2*width), 'relu'),
	]

	for name, var, act in strategies:
		np.random.seed(42)  # same init for fair comparison
		stds = propagate_through_layers(n_layers, layer_sizes, var, act)

		print(f"\n  {name}  (activation={act})")
		print(f"  {'Layer':>6} | {'Std Dev':>14} | {'Status':>15}")
		print(f"  {'-'*42}")
		for i, s in enumerate(stds):
			if s > 1e10:
				status = "EXPLODED"
			elif s < 1e-5:
				status = "VANISHED"
			elif 0.5 < s < 2.0:
				status = "Stable"
			else:
				status = "Drifting"
			if i <= 5 or i == len(stds) - 1 or i % 3 == 0:
				print(f"  {i:6d} | {s:>14.6f} | {status:>15}")
		time.sleep(0.3)

	# Q: Only He initialization keeps activations stable through all 15 layers
	# Q: Xavier works for tanh/sigmoid but not ReLU (ReLU kills half the variance)
	# Q: What happens with 50 layers? 100 layers?


def xavier_vs_he():
	"""
	Show that Xavier works for tanh but not ReLU,
	and He works for ReLU but overestimates for tanh.
	"""
	print(f"\n{'='*70}")
	print(f"Xavier vs He: Matching Initialization to Activation")
	print(f"{'='*70}")

	n_layers = 15
	width = 256
	layer_sizes = [width] * (n_layers + 1)

	combos = [
		('Xavier + tanh',  2.0 / (2*width), 'tanh'),
		('Xavier + ReLU',  2.0 / (2*width), 'relu'),
		('He + ReLU',      2.0 / width,     'relu'),
		('He + tanh',      2.0 / width,     'tanh'),
	]

	print(f"\n  {'Combination':<25} {'Std at Layer 0':>14} {'Std at Layer 15':>16} {'Verdict':>10}")
	print(f"  {'-'*70}")

	for name, var, act in combos:
		np.random.seed(42)
		stds = propagate_through_layers(n_layers, layer_sizes, var, act)
		s_first = stds[0]
		s_last = stds[-1]
		if 0.3 < s_last < 3.0:
			verdict = "Good"
		elif s_last < 0.01:
			verdict = "Vanishing"
		elif s_last > 100:
			verdict = "Exploding"
		else:
			verdict = "Drifting"
		print(f"  {name:<25} {s_first:>14.6f} {s_last:>16.6f} {verdict:>10}")
		time.sleep(0.3)

	# Q: Xavier + tanh: good.  Xavier + ReLU: vanishing.
	# Q: He + ReLU: good.     He + tanh: exploding.
	# Q: Match your initialization to your activation function!


def batch_norm_demo():
	"""
	Demonstrate how batch normalization rescues bad initialization.

	BatchNorm: x_hat = (x - mu_B) / sqrt(sigma_B^2 + epsilon)

	The epsilon prevents division by zero when all inputs are identical.
	"""
	print(f"\n{'='*70}")
	print(f"Batch Normalization: Rescuing Bad Initialization")
	print(f"{'='*70}")

	n_layers = 15
	width = 256
	batch_size = 64
	eps = 1e-5

	# Bad initialization: Var = 1.0 (way too large for ReLU)
	init_var = 1.0

	print(f"\n  Using bad init (Var=1.0) with and without BatchNorm")
	print(f"  {'Layer':>6} | {'Without BN':>14} | {'With BN':>14}")
	print(f"  {'-'*40}")

	np.random.seed(42)
	x_no_bn = np.random.randn(batch_size, width)
	x_bn = x_no_bn.copy()

	for layer in range(n_layers):
		W = np.random.randn(width, width) * np.sqrt(init_var)

		# Without BatchNorm
		x_no_bn = x_no_bn @ W
		x_no_bn = np.maximum(0, x_no_bn)  # ReLU

		# With BatchNorm
		x_bn = x_bn @ W
		# BatchNorm: normalize to zero mean, unit variance
		mu = np.mean(x_bn, axis=0)
		var = np.var(x_bn, axis=0)
		x_bn = (x_bn - mu) / np.sqrt(var + eps)
		x_bn = np.maximum(0, x_bn)  # ReLU

		std_no_bn = np.std(x_no_bn)
		std_bn = np.std(x_bn)
		if layer <= 5 or layer == n_layers - 1 or layer % 3 == 0:
			print(f"  {layer:6d} | {std_no_bn:>14.6f} | {std_bn:>14.6f}")

	print(f"\n  BatchNorm keeps activations stable regardless of initialization!")
	print(f"  Q: Why is epsilon needed? What if all inputs in a batch are identical?")
	print(f"  Q: What epsilon value is too large? When would it hurt?")


if __name__ == "__main__":

	# =================================================================
	# Configuration
	# =================================================================

	np.random.seed(42)

	# =================================================================

	initialization_experiment()
	xavier_vs_he()
	batch_norm_demo()
