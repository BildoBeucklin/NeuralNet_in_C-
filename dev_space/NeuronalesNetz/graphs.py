from numpy import loadtxt
import matplotlib.pyplot as plt

t = []
timeimprovOMP = []
timeimprovSIMD = []

dtimeNP = loadtxt('timeNP', delimiter=',')
dtimeOMP = loadtxt('timeOMP', delimiter=',')
dtimeSIMD = loadtxt('timeSIMD', delimiter=',')

accuracyNP = loadtxt('accuracyNP', delimiter=',')
accuracyOMP = loadtxt('accuracyOMP', delimiter=',')
accuracySIMD = loadtxt('accuracySIMD', delimiter=',')

for i in range(0, len(dtimeNP)):
    timeimprovOMP.append(dtimeNP[i]/dtimeOMP[i])
    timeimprovSIMD.append(dtimeNP[i]/dtimeSIMD[i])
    t.append(i)

fig, (ax1, ax2, ax3) = plt.subplots(3)
ax1.plot(t,dtimeNP, "-b", label="No parallelization")
ax1.plot(t,dtimeOMP,"-r", label="omp")
ax1.plot(t,dtimeSIMD,"-g", label="simd")
ax1.set_title('Runtime of different parallelization')
ax1.set(xlabel = 'Epochs', ylabel='Time in s')
ax1.legend(loc='best')

ax2.plot(t,accuracyNP, "-b", label="No parallelization")
ax2.plot(t,accuracyOMP,"-r", label="omp")
ax2.plot(t,accuracySIMD,"-g", label="simd")
ax2.set_title('Accuracy of different parallelization')
ax2.set(xlabel = 'Epochs', ylabel='Accuracy in %')
ax2.legend(loc='best')

ax3.plot(t,timeimprovOMP, "-b", label="OMP")
ax3.plot(t,timeimprovSIMD,"-r", label="SIMD")
ax3.set_title('Speedup compared to no parallelization')
ax3.set(xlabel = 'Epochs', ylabel='Speedup')
ax3.legend(loc='best')

fig.tight_layout()
fig.savefig("all_graphs.png",dpi=600)
fig.show()

fig1, ax = plt.subplots(1)
ax.plot(t,dtimeNP, "-b", label="No parallelization")
ax.plot(t,dtimeOMP,"-r", label="omp")
ax.plot(t,dtimeSIMD,"-g", label="simd")
ax.set_title('Runtime of different parallelization')
ax.set(xlabel = 'Epochs', ylabel='Time in s')
ax.legend(loc='best')
fig1.savefig("runtime_single.png",dpi=600)

fig2, ax4 = plt.subplots(1)
ax4.plot(t,accuracyNP, "-b", label="No parallelization")
ax4.plot(t,accuracyOMP,"-r", label="omp")
ax4.plot(t,accuracySIMD,"-g", label="simd")
ax4.set_title('Accuracy of different parallelization')
ax4.set(xlabel = 'Epochs', ylabel='Accuracy in %')
ax4.legend(loc='best')
fig2.savefig("accuracy_single.png",dpi=600)

fig3, ax5 = plt.subplots(1)
ax5.plot(t,timeimprovOMP, "-b", label="omp")
ax5.plot(t,timeimprovSIMD,"-r", label="simd")
ax5.set_title('Speedup compared to no parallelization')
ax5.set(xlabel = 'Epochs', ylabel='Speedup')
ax5.legend(loc='best')
fig3.savefig("speedup_single.png",dpi=600)
