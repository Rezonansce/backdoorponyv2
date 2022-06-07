import matplotlib.backends.backend_pdf
import matplotlib.pyplot as plt
import numpy as np

pdf = matplotlib.backends.backend_pdf.PdfPages('output.pdf')

benign = [[97.84, 97.42, 97.48, 97.08, 96.38], [95.42, 95.95, 94.81, 95.89, 96.79]]
fp = [[0.32, 0.47, 0.39], [0.0, 0.5, 0.93], [0.06, 0.12, 0.23], [0.01, 0.17, 0.92], [0.0, 0.08, 0.17], [0.0, 0.04, 0.13], [2.22, 0.84, 1.56], [0.07, 0.85, 2.22], [1.98, 0.25, 2.13], [1.37, 0.82, 2.56]]
poison = [[99.74, 100.0, 100.0, 99.9, 100.0], [35.39, 89.8, 97.19, 97.34, 100.0]]
tp = [[0.0, 5.83, 29.91], [0.0, 0.0, 5.78], [0.02, 0.04, 0.06], [0.0, 0.0, 0.0], [0.0, 0.1, 0.2], [0.0, 0.0, 0.09], [1.54, 0.12, 0.29], [0.04, 0.31, 0.94], [1.71, 0.13, 1.35], [1.76, 1.29, 3.16]]

# Change these results with the more complex set
results = {'benign': [[96.8, 96.13], [97.28, 98.01]], 'fp': [[0.64, 0.15], [0.0, 0.06], [0.02, 0.1], [0.0, 0.03]], 'poison': [[100.0, 100.0], [100.0, 99.91]], 'tp': [[0.0, 24.17], [0.0, 0.9], [0.0, 23.22], [0.0, 0.0]]}
results_benign = results['benign']
benign_pattern_10 = results_benign[0][0]
benign_pattern_25 = results_benign[0][1]
benign_pixel_10 = results_benign[1][0]
benign_pixel_25 = results_benign[1][1]


params = ('10%', '25%', '33%', '50%', '75%', '80%')
y_pos = np.arange(len(params))

# Here add entries like benign_patten_10, benign_pattern_25 instead of 99, 98
# Then adjust scale
performance = [99,98,96,94,92,91]

plt.figure()
#plt.bar(y_pos, performance, align='center', alpha=0.5)
plt.plot(y_pos, performance)
plt.xticks(y_pos, params)
plt.ylabel('Accuracy')  
plt.xlabel('Percentage of poisoned data')
plt.title('Accuracy on benign data with pattern trigger')
plt.ylim(90,100)
plt.show()
plt.savefig(pdf, format='pdf')

plt.figure()
plt.bar(y_pos, performance, align='center', alpha=0.5)
#plt.plot(y_pos, performance)
plt.xticks(y_pos, params)
plt.ylabel('Accuracy2')  
plt.xlabel('Percentage of poisoned data2')
plt.title('Accuracy on benign data with pattern trigger2')
plt.ylim(90,100)
plt.show()
plt.savefig(pdf, format='pdf')


#new
plt.figure()
plt.bar((10, 25, 33, 50, 75), benign[1], align='center', alpha=0.5, width=5)
plt.ylabel('Accuracy')  
plt.xlabel('Percentage of poisoned data')
plt.title('Accuracy on benign data with pattern trigger')
plt.ylim(90,100)
plt.savefig(pdf, format='pdf')

plt.figure()
plt.bar((10, 25, 33, 50, 75), poison[1], align='center', alpha=0., width=5)
plt.ylabel('Accuracy')  
plt.xlabel('Percentage of poisoned data')
plt.title('Accuracy on poisoned data with pattern trigger')
plt.savefig(pdf, format='pdf')

plt.figure()
plt.bar((10, 25, 33, 50, 75), benign[1], align='center', alpha=0.5, width=5)
plt.ylabel('Accuracy')  
plt.xlabel('Percentage of poisoned data')
plt.title('Accuracy on benign data with pixel trigger')
plt.ylim(90,100)
plt.savefig(pdf, format='pdf')

plt.figure()
plt.bar((10, 25, 33, 50, 75), poison[1], align='center', alpha=0.5, width=5)
plt.ylabel('Accuracy')  
plt.xlabel('Percentage of poisoned data')
plt.title('Accuracy on poisoned data with pixel trigger')
plt.savefig(pdf, format='pdf')

i=0
for x in (10, 25, 33, 50, 75):
    ++i
    plt.figure()
    plt.bar((10, 50, 100), fp[i], align='center', alpha=0.5, width=20)
    plt.ylabel('Clean images used in STRIP')  
    plt.xlabel('Percentage of false positives')
    plt.title('False poitives on ' + str(x) + '% poison')
    plt.savefig(pdf, format='pdf')

    plt.figure()
    plt.bar((10, 50, 100), tp[i], align='center', alpha=0.5, width=20)
    plt.ylabel('Clean images used in STRIP')  
    plt.xlabel('Percentage of true positives')
    plt.title('True poitives on ' + str(x) + '% poison')
    plt.savefig(pdf, format='pdf')

pdf.close()



