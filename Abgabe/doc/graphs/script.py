#TODO: zeit improvement plotten(simd, omp ,normal)
#TODO: genauigkeit mit steigung der epochs plotten
#TODO:vlt Live Demo image to csv
import subprocess
import time
from numpy import savetxt

t = []
accuracyNP = []
accuracyOMP = []
accuracySIMD = []
epochs = 5
dtimeNP = []
dtimeOMP = []
dtimeSIMD = []
learnStep = 0.01
for i in range(1,epochs):
    time_start = time.time()
    p= subprocess.run(["./net-exe",str(i),str(learnStep), "20", "10000", "60000", "3", "784", "16", "10" ], stdout=subprocess.PIPE)
    time_end = time.time()
    output= float(p.stdout.decode())
    accuracyNP.append(output)
    dtimeNP.append(time_end-time_start)
    t.append(i)
    print( "Learn-step:",learnStep,"\nEpochs:",i,"\nAccuracy:",output,"%\nTime:",(time_end-time_start),"s\n")



savetxt('accuracy',accuracyNP,delimiter=',')
savetxt('time',dtimeNP,delimiter=',')


