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
epochs = 100
dtimeNP = []
dtimeOMP = []
dtimeSIMD = []
learnStep = 0.1
for i in range(1,epochs):
    time_start = time.time()
    p= subprocess.run(["./neuronet-deltalearn",str(i),str(learnStep)], stdout=subprocess.PIPE)
    time_end = time.time()
    output= float(p.stdout.decode())
    accuracyNP.append(output)
    dtimeNP.append(time_end-time_start)
    t.append(i)
    print( "Learn-step:",learnStep,"\nEpochs:",i,"\nAccuracy:",output,"%\nTime:",(time_end-time_start),"s\n")

for i in range(1,epochs):
    time_start = time.time()
    p= subprocess.run(["./neuronet-deltalearn-parallel",str(i),str(learnStep)],cwd='./omp', stdout=subprocess.PIPE)
    time_end = time.time()
    output= float(p.stdout.decode())
    accuracyOMP.append(output)
    dtimeOMP.append(time_end-time_start)
    print( "Learn-step:",learnStep,"\nEpochs:",i,"\nAccuracy:",output,"%\nTime:",(time_end-time_start),"s\n")

for i in range(1, epochs):
    time_start = time.time()
    p = subprocess.run(["./neuronet-deltalearn-simd",str(i),str(learnStep)], cwd='./simd', stdout=subprocess.PIPE)
    time_end = time.time()
    output= float(p.stdout.decode())
    accuracySIMD.append(output)
    dtimeSIMD.append(time_end-time_start)
    print( "Learn-step:",learnStep,"\nEpochs:",i,"\nAccuracy:",output,"%\nTime:",(time_end-time_start),"s\n")


savetxt('accuracyNP',accuracyNP,delimiter=',')
savetxt('accuracyOMP',accuracyOMP,delimiter=',')
savetxt('accuracySIMD',accuracySIMD,delimiter=',')
savetxt('timeNP',dtimeNP,delimiter=',')
savetxt('timeOMP',dtimeOMP,delimiter=',')
savetxt('timeSIMD',dtimeSIMD,delimiter=',')

