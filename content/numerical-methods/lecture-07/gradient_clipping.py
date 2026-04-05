# Gradient Clipping: Preventing Exploding Gradients
#
# When gradients become extremely large, parameter updates overshoot
# and training becomes unstable (loss spikes, NaN values).
#
# Two approaches:
#   Clip by Value:  g'_i = clip(g_i, -tau, tau)       (changes direction!)
#   Clip by Norm:   g' = g * min(1, tau / ||g||)       (preserves direction)
#
# Clip by norm is preferred because it preserves the gradient direction.

import time
import numpy as np


def clip_by_value(gradient, tau):
	"""
	Clip each gradient component independently to [-tau, tau].

	Simple but dangerous: can drastically change gradient direction.
	Example: g = [1000, 1], tau = 1  ->  g' = [1, 1]
	         Direction changed from nearly horizontal to 45 degrees!
	"""
	return np.clip(gradient, -tau, tau)


def clip_by_norm(gradient, tau):
	"""
	Scale the entire gradient vector if its norm exceeds tau.

	g' = g * min(1, tau / ||g||)

	Properties:
	  - Direction is preserved (g' points the same way as g)
	  - Norm is bounded: ||g'|| <= tau
	  - Small gradients are unchanged: if ||g|| <= tau, g' = g
	"""
	norm = np.linalg.norm(gradient)
	if norm > tau:
		gradient = gradient * (tau / norm)
	return gradient


def exploding_gradients_demo():
	"""
	Simulate gradient explosion in a deep network.

	In an RNN or deep network, gradients flow through T layers via:
	   dL/dh_0 = dL/dh_T * prod(W^T)

	If ||W|| > 1, gradients grow as ||W||^T -- exponentially!
	"""
	print(f"\n{'='*70}")
	print(f"Gradient Explosion in Deep Networks")
	print(f"{'='*70}")
	print(f"{'Depth T':>8} | {'||grad||':>14} | {'Status':>20}")
	print(f"{'-'*50}")

	# Simulate a weight matrix with spectral norm slightly > 1
	W_norm = 1.5   # ||W|| > 1 causes explosion
	grad_norm = 1.0

	for T in [1, 2, 5, 10, 20, 50, 100]:
		exploded_norm = grad_norm * (W_norm ** T)
		if exploded_norm > 1e15:
			status = "OVERFLOW DANGER"
		elif exploded_norm > 1e6:
			status = "Exploding"
		elif exploded_norm > 100:
			status = "Growing fast"
		else:
			status = "Manageable"
		print(f"{T:8d} | {exploded_norm:>14.2e} | {status:>20}")
		time.sleep(0.3)

	print(f"\n  With ||W|| = {W_norm}, gradients grow as {W_norm}^T")
	print(f"  At T=100: gradient is {W_norm**100:.2e} times the original!")
	print(f"  Q: What if ||W|| = 0.9 instead? (Hint: vanishing gradients)")


def clipping_comparison():
	"""
	Compare clip-by-value and clip-by-norm on the same gradient.
	"""
	print(f"\n{'='*70}")
	print(f"Gradient Clipping: Value vs Norm")
	print(f"{'='*70}")

	test_cases = [
		(np.array([3.0, 4.0]),         2.0, "g=[3,4], tau=2"),
		(np.array([1000.0, 1.0]),      1.0, "g=[1000,1], tau=1"),
		(np.array([0.5, 0.3]),         2.0, "g=[0.5,0.3], tau=2 (no clip needed)"),
		(np.array([100.0, 100.0]),     5.0, "g=[100,100], tau=5"),
		(np.array([-3.0, 4.0, 0.0]),  2.0, "g=[-3,4,0], tau=2"),
	]

	for g, tau, label in test_cases:
		g_val = clip_by_value(g.copy(), tau)
		g_norm = clip_by_norm(g.copy(), tau)

		print(f"\n  {label}")
		print(f"    Original:     {g}  ||g|| = {np.linalg.norm(g):.4f}")
		print(f"    Clip by value: {g_val}  ||g'|| = {np.linalg.norm(g_val):.4f}")
		print(f"    Clip by norm:  {g_norm}  ||g'|| = {np.linalg.norm(g_norm):.4f}")

		# Show direction change
		if np.linalg.norm(g) > 0 and np.linalg.norm(g_val) > 0:
			cos_val = np.dot(g, g_val) / (np.linalg.norm(g) * np.linalg.norm(g_val))
			cos_norm_clip = np.dot(g, g_norm) / (np.linalg.norm(g) * np.linalg.norm(g_norm))
			print(f"    Direction preserved? value: cos={cos_val:.4f}  norm: cos={cos_norm_clip:.4f}")
		time.sleep(0.3)

	# Q: For g=[1000,1], clip-by-value gives [1,1] -- direction changed from
	#    nearly horizontal to 45 degrees. Clip-by-norm preserves direction.
	# Q: When ||g|| <= tau, both methods leave the gradient unchanged.


def sgd_with_clipping():
	"""
	Simple SGD loop showing the effect of gradient clipping.
	Minimizes f(x) = x^4 which has a very steep gradient for large x.
	"""
	print(f"\n{'='*70}")
	print(f"SGD with/without Gradient Clipping: f(x) = x^4")
	print(f"{'='*70}")

	f = lambda x: x**4
	grad_f = lambda x: 4 * x**3

	x0 = 5.0
	lr = 0.001
	tau = 10.0
	n_steps = 20

	print(f"  x0={x0}, lr={lr}, tau={tau}")
	print(f"\n{'Step':>4} | {'x (no clip)':>14} | {'grad':>12} | {'x (clipped)':>14} | {'clipped grad':>12}")
	print(f"{'-'*70}")

	x_noclip = x0
	x_clipped = x0

	for t in range(n_steps):
		g_noclip = grad_f(x_noclip)
		g_clip = grad_f(x_clipped)
		g_clip_applied = np.clip(g_clip, -tau, tau)  # clip by value for simplicity

		x_noclip = x_noclip - lr * g_noclip
		x_clipped = x_clipped - lr * g_clip_applied

		if t < 10 or t % 5 == 0:
			print(f"{t:4d} | {x_noclip:>14.6f} | {g_noclip:>12.2f} | {x_clipped:>14.6f} | {g_clip_applied:>12.2f}")
		time.sleep(0.2)

	print(f"\n  Without clipping: final x = {x_noclip:.6f}  f(x) = {f(x_noclip):.6f}")
	print(f"  With clipping:    final x = {x_clipped:.6f}  f(x) = {f(x_clipped):.6f}")

	# Q: Does clipping slow down convergence? When is it worth it?
	# Q: What happens if tau is too small? Too large?


if __name__ == "__main__":
	exploding_gradients_demo()
	clipping_comparison()
	sgd_with_clipping()
