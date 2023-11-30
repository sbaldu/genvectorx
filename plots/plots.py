from array import array
import subprocess
import math
import os
import sys
import numpy as np

buildcuda = '/home/mdessole/Projects/ROOT/genvectorxfork/build_cuda/'
buildoneapi = '/home/mdessole/Projects/ROOT/genvectorxfork/build_oneapi/'
buildacpp = '/home/mdessole/Projects/ROOT/genvectorxfork/build_adaptivecpp/'
path = '/home/mdessole/Projects/ROOT/genvectorxfork/plots/'

def result2list(exe, N, nruns):
    print(exe, str(int(N)), str(int(nruns)))
    result = subprocess.run([exe, str(int(N)), str(int(nruns)) ], stdout=subprocess.PIPE).stdout.decode('utf-8')
    print(result)
    if len(result)<=0:
        return []
    result = result.replace("\r","").replace("\n","")
    time = np.zeros(nruns)
    for r in range(nruns):
        i = result.find("time")
        j = result.find("(s)")
        if (i<0) or (j<0):
            break
        try:
            time[r] = float(result[i+5:j])
        except:
            time[r] = 0.
        try:
            result = result[j+3:]
        except:
            result = ""
    #endif
    return time

def run_benchmark(builddir, sizes, nruns = 1, testname = "InvariantMasses", btype = 'cpu'):
    timeopt = array( 'd' )
    timestd = array( 'd' )
    
    if btype == 'cpu':
        exe = os.path.join(builddir,'testx',testname)
    elif btype == 'cuda':
        exe = os.path.join(builddir,'testx',testname+'CUDA')
    elif btype == 'sycl':
        exe = os.path.join(builddir,'testx',testname+'SYCL')
    #endif
    
    if (not os.path.exists(exe)):
        print("Executable", exe, "not found!")
        return timeopt, timestd
    
    for i, N in enumerate(sizes):
        time = result2list(exe,N,nruns)
        if (len(time)<=0):
            return timeopt, timestd
        print(time, '\n')
        timeopt.append(np.amin(time))
        timestd.append(np.std(time)) 
    #endfor
    return timeopt, timestd


def collect_results(testname, platform, nruns):
    nruns = int(nruns)

    sizes = array('d',[1000, 10000, 100000,1000000,5000000,10000000, 50000000,100000000]) #10,100,1000,10000,100000,
    Ssizes = array('d',[1000, 10000, 100000,1000000,5000000,10000000, 50000000,100000000,200000000]) #10,100,1000,10000,100000,

    val = list(sizes)
    sizes_gb = [v*8*4/1e9 for v in val] #(nb of doubles)*(bytes for doubles)*(4=dim of LVector)/(bytes in GB)
    sizes_gb = array('d', sizes_gb)

    val = list(Ssizes)
    Ssizes_gb = [v*4*4/1e9 for v in val] #(nb of floats)*(bytes for floats)*(4=dim of LVector)/(bytes in GB)
    Ssizes_gb = array('d', Ssizes_gb)

    savez_dict = {}
    savez_dict['sizes'] = sizes 
    savez_dict['Ssizes'] = Ssizes 
    savez_dict['sizes_gb'] = sizes_gb
    savez_dict['Ssizes_gb'] = Ssizes_gb 
    
    if (platform == 'cpu'):
        timecpu, stdcpu = run_benchmark(buildacpp, sizes, testname = testname, nruns = nruns, btype = 'cpu')
        Stimecpu, Sstdcpu = run_benchmark(buildacpp, Ssizes, testname = "S"+testname, nruns = nruns, btype = 'cpu')
        savez_dict['timecpu'] = timecpu
        savez_dict['Stimecpu'] = Stimecpu
        savez_dict['stdcpu'] = stdcpu
        savez_dict['Sstdcpu'] = Sstdcpu
    else:
        if (platform != "hip"):
            timeoneapi, stdoneapi = run_benchmark(buildoneapi, sizes, testname = testname, nruns = nruns, btype = 'sycl')
            Stimeoneapi, Sstdoneapi = run_benchmark(buildoneapi, Ssizes, testname = "S"+testname, nruns = nruns, btype = 'sycl')
            savez_dict['timeoneapi'] = timeoneapi
            savez_dict['Stimeoneapi'] = Stimeoneapi
            savez_dict['stdoneapi'] = stdoneapi
            savez_dict['Sstdoneapi'] = Sstdoneapi
        #endif

        timeacpp, stdacpp = run_benchmark(buildacpp, sizes, testname = testname, nruns = nruns, btype = 'sycl')
        Stimeacpp, Sstdacpp = run_benchmark(buildacpp, Ssizes, testname = "S"+testname, nruns = nruns, btype = 'sycl')
        savez_dict['timeacpp'] = timeacpp
        savez_dict['Stimeacpp'] = Stimeacpp
        savez_dict['stdacpp'] = stdacpp
        savez_dict['Sstdacpp'] = Sstdacpp

        if (platform == "cuda"):
            timecuda, stdcuda = run_benchmark(buildcuda, sizes, testname = testname, nruns = nruns, btype = 'cuda')
            Stimecuda, Sstdcuda = run_benchmark(buildcuda, Ssizes, nruns = nruns, testname = "S"+testname, btype = 'cuda')
            savez_dict['timecuda'] = timecuda
            savez_dict['Stimecuda'] = Stimecuda
            savez_dict['stdcuda'] = stdcuda
            savez_dict['Sstdcuda'] = Sstdcuda
        #endif
    #endif
    np.savez(path+testname + "_" + platform + "_nruns" + str(nruns) + ".npz", **savez_dict)
    return

if __name__ == "__main__":
    collect_results(sys.argv[1],sys.argv[2], sys.argv[3])

