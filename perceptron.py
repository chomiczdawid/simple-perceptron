import pandas as pd
import matplotlib.pyplot as plt
from random import random, seed


data = pd.read_csv('exercise1.csv', sep=';')  # Can change to exercise2 or exercise3
print(data)  # Print for analysing the algorithm

# Initiation
weight_1 = random()
weight_2 = random()
bias = random()
score = 0

print('\nPrimary weights: [{}, {}, {}]'.format(bias, weight_1, weight_2))

while score < len(data):
    # Calculation
    result = weight_1 * data['x'] + weight_2 * data['y'] + bias
    print('\nResult:\n{}'.format(result))  # Print for analysing the algorithm
    # Validation
    print('\nValidation:')
    score = 0
    for i in range(len(result)):
        if result[i] > 0:
            color = 1
        else:
            color = 0
        error = data['target'][i] - color
        print('Row {} error = {}'.format(i, error))  # Print for analysing the algorithm
        if error == 0:
            score += 1
        else:
            bias += error
            weight_1 += error * data['x'][i]
            weight_2 += error * data['y'][i]
            print('\nNew weights: [{}, {}, {}]'.format(bias, weight_1, weight_2))  # Print for analysing the algorithm
            break

print('\nSuccessful classification!'
      '\nWeight vector: [{}, {}, {}]\n'.format(round(bias,2),round(weight_1,2),round(weight_2,2)))


# Visualisation
x = data['x']
slope = -(bias/weight_2)/(bias/weight_1)
intercept = -bias/weight_2

plt.plot(x, slope * x + intercept)
plt.xlim(-3,3)
plt.ylim(-3,3)

groups = data.groupby("target")
for name, group in groups:
    plt.plot(group['x'], group['y'], marker="o", linestyle="", label=name)

plt.legend()
plt.show()