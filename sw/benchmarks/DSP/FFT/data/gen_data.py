#!/usr/bin/env python3

import numpy as np
import sys

def rand_matrix(N, M, seed):
	return np.random.normal(size=(N, M)).astype(np.float64)
	# return np.arange(seed, seed+N*M, dtype=np.float64).reshape(N, M) * 3.141
	# return np.random.randn(N*M).astype(np.float64).reshape(N, M)

def twiddle(N):
	v = np.exp(-2j * np.pi * np.arange(N//2) / N)
	v = v.astype(np.complex128)
	return np.array((np.real(v), np.imag(v))).transpose()

def complex_mul(a, b):
	return np.array((
		a[:,0]*b[:,0] - a[:,1]*b[:,1],
		a[:,0]*b[:,1] + a[:,1]*b[:,0],
	)).transpose()

def fft(x, twiddle):
	if x.shape[0] == 1:
		return x
	else:
		E = fft(x[0::2,:], twiddle)
		O = fft(x[1::2,:], twiddle)
		tw = twiddle(x.shape[0])
		twO = complex_mul(tw, O)
		return np.concatenate((
			E + twO,
			E - twO
		), axis=0)

N = 128
x = rand_matrix(N, 2, 1)
# sys.stderr.write("x:\n%s\n" % x)
x_fused = x[:,0] + 1j * x[:,1]
# random_twiddle = rand_matrix(N//2, 2, 2)
y = fft(x, twiddle)
# y = fft(x, lambda i: random_twiddle[0::N//i])
y_baseline = np.fft.fft(x_fused).astype(np.complex128)
# sys.stderr.write("y:\n%s\n" % y)
# sys.stderr.write("y_baseline:\n%s\n" % y_baseline)

field = y.reshape(2*N)
checksum = np.array((
	np.sum(field[0::4], axis=0),
	np.sum(field[1::4], axis=0),
	np.sum(field[2::4], axis=0),
	np.sum(field[3::4], axis=0),
))
# sys.stderr.write("checksum:\n%s\n" % checksum)

def emit(name, array):
	print(".global %s" % name)
	print(".align 3")
	print("%s:" % name)
	bs = array.tobytes()
	for i in range(0, len(bs), 4):
		s = ""
		for n in range(4):
			s += "%02x" % bs[i+3-n]
		print("    .word 0x%s" % s)

print(".section .l1,\"aw\",@progbits")
emit("input_size", np.array(N, dtype=np.uint32))
emit("input", x)
emit("input_twiddle", twiddle(N))
emit("output", y)
# emit("output_checksum", checksum)
