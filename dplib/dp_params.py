
""" Calculation of (epislon, delta) based on desired generalization guarantees.
The paper https://arxiv.org/pdf/1511.02513.pdf contains theory regarding relation 
between (epsilon, delta) as differential privacy parameters and (alpha, beta) as 
generalization parameters."""

from math import sqrt

def calculate_dp_params(n, k ,alpha = None, beta = None, f1 = 1, f2 = 1 ):
	if (alpha is not None) and (beta is not None):
		epsilon = f1*alpha*1.0/sqrt(k)
	  	delta = f2*beta*1.0/k
	return (epsilon, delta)