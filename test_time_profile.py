from __future__ import print_function
import problems.real as prb
import matplotlib.pyplot as plt

# code used to test correctness of inflow time profiles used in real.py

# print(prb.Problem.v_function_2(0.9375*0.3334))
# print(prb.Problem.v_function_2(0.9375*0.3333))

dt = 0.001
end = 2.1
rng = []
t = 0.
while t < end+0.00001:
    rng.append(t)
    t += dt

# for f in [prb.Problem.v_function, prb.Problem.v_function_2]:
for f in [prb.Problem.v_function_2]:
    Tvalues = []
    Tx = []
    for t in rng:
        Tvalues.append(f(t))
        Tx.append(t)
    plt.plot(Tx, Tvalues)
plt.show()