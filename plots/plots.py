from array import array
import subprocess
import math
import os
import sys
import time
import numpy as np
from pathlib import Path

buildcuda = '/home/mdessole/Projects/GenVectorX/genvectorx/build_cuda/'
buildoneapi = '/home/mdessole/Projects/GenVectorX/genvectorx/build_oneapi/'
buildacpp = '/home/mdessole/Projects/GenVectorX/genvectorx/build_adaptivecpp/'
path = '/home/mdessole/Projects/GenVectorX/genvectorx/plots/'
nsys = "/usr/local/cuda-12.2/bin/nsys"

base_header = "N,env,type,testname,precision"

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

def run_benchmark(builddir, sizes, nruns = 1,                         
                testname = "InvariantMasses", 
                precision = 64, btype = 'cpu'):
    timeopt = array( 'd' )
    timestd = array( 'd' )
    
    if precision == 32:
        testname = 'S'+testname
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

def run_nsys_benchmark(builddir, sizes, nruns = 1, 
                        testname = "InvariantMasses", 
                        memtype = '',
                        precision = 64,
                        btype = 'cpu', 
                        output_file = 'output_file'):

    if precision == 32:
        testname = 'S'+testname
    if btype == 'cpu':
        exe = os.path.join(builddir,'testx',testname)
        environment = 'CPU'
    elif btype == 'cuda':
        exe = os.path.join(builddir,'testx',testname+'CUDA')
        environment = 'CUDA'
    elif btype == 'sycl':
        exe = os.path.join(builddir,'testx',testname+'SYCL')
        if "oneapi" in builddir.lower():
            environment = 'oneAPI'
        if ("acpp" in builddir.lower()) or ("adaptivecpp" in builddir.lower()):
            environment = 'AdaptiveCPP'
    else:
        raise ValueError('Unknown backend')
    #endif
    if precision == 32:
        testname = testname[1:]
    
    if (not os.path.exists(exe)):
        print("Executable", exe, "not found!")
        return 

            
    if not os.path.exists(f"{output_file}"):
        os.mkdir(f"{output_file}")
    
    for i, N in enumerate(sizes):
        time = result2list(exe,N,nruns)
        arg = f"{int(N)} {nruns}"
        # Profile code with nsys
        try:
            profile = subprocess.run(
                                    [nsys, "profile", "-otemp", exe, *arg.split()],
                                    #env={**os.environ, e: "1"},
                                    stdout=subprocess.PIPE,
                                    check=True,
                            )
        except subprocess.CalledProcessError:
            continue


        # Gather statistics
        result = subprocess.run(
            [nsys, "stats", "temp.nsys-rep", "--format=csv"],
            stdout=subprocess.PIPE,
            check=True
        )

        subprocess.run(["rm", "temp.nsys-rep", "temp.sqlite"])
        output = result.stdout.decode("utf-8").split("\n\n")
        
        # If output file does not exist, write headers
        if not Path(f"{output_file}/api").exists():
            with open(f"{output_file}/api", "w") as file_handler:
                header = output[2].split("\n")[1]
                if 'SKIPPED' in header:
                    print(output)
                    quit()
                file_handler.write(f"{base_header},{header}\n")
        if not Path(f"{output_file}/kernel").exists():
            with open(f"{output_file}/kernel", "w") as file_handler:
                header = output[3].split("\n")[1]
                file_handler.write(f"{base_header},{header}\n")
        if not Path(f"{output_file}/memop").exists():
            with open(f"{output_file}/memop","w") as file_handler:
                header = output[4].split("\n")[1]
                file_handler.write(f"{base_header},{header}\n")

        # Write row
        out_base = f"{int(N)},{environment},{memtype},{testname},{precision}"
        with open(
            f"{output_file}/api",
            "a",
        ) as file_handler:
            file_handler.write(
                "\n".join([f"{out_base},{s}" for s in output[2].split("\n")[2:]])
            )
            file_handler.write("\n")
        with open(
            f"{output_file}/kernel",
            "a",
        ) as file_handler:
            file_handler.write(
                "\n".join([f"{out_base},{s}" for s in output[3].split("\n")[2:]])
            )
            file_handler.write("\n")
        with open(
            f"{output_file}/memop",
            "a",
        ) as file_handler:
            file_handler.write(
                "\n".join([f"{out_base},{s}" for s in output[4].split("\n")[2:]])
            )
            file_handler.write("\n")

        # with open(f"{outpu}/api/N{int(N)}", "w") as file_handler:
        #     file_handler.write("\n".join(output[2].split("\n")[1:]))
        # with open(f"{output_file}/kernel/N{int(N)}", "w") as file_handler:
        #     file_handler.write("\n".join(output[3].split("\n")[1:]))
        # with open(f"{output_file}/memop/N{int(N)}", "w") as file_handler:
        #     file_handler.write("\n".join(output[4].split("\n")[1:]))
    #endfor
    return 

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
        Stimecpu, Sstdcpu = run_benchmark(buildacpp, Ssizes, testname = testname, precision = 32, nruns = nruns, btype = 'cpu')
        savez_dict['timecpu'] = timecpu
        savez_dict['Stimecpu'] = Stimecpu
        savez_dict['stdcpu'] = stdcpu
        savez_dict['Sstdcpu'] = Sstdcpu
    else:
        if (platform != "hip"):
            timeoneapi, stdoneapi = run_benchmark(buildoneapi, sizes, testname = testname, nruns = nruns, btype = 'sycl')
            Stimeoneapi, Sstdoneapi = run_benchmark(buildoneapi, Ssizes, testname = testname, precision = 32, nruns = nruns, btype = 'sycl')
            savez_dict['timeoneapi'] = timeoneapi
            savez_dict['Stimeoneapi'] = Stimeoneapi
            savez_dict['stdoneapi'] = stdoneapi
            savez_dict['Sstdoneapi'] = Sstdoneapi
        #endif

        timeacpp, stdacpp = run_benchmark(buildacpp, sizes, testname = testname, nruns = nruns, btype = 'sycl')
        Stimeacpp, Sstdacpp = run_benchmark(buildacpp, Ssizes, testname = testname, precision = 32, nruns = nruns, btype = 'sycl')
        savez_dict['timeacpp'] = timeacpp
        savez_dict['Stimeacpp'] = Stimeacpp
        savez_dict['stdacpp'] = stdacpp
        savez_dict['Sstdacpp'] = Sstdacpp

        if ("cuda" in platform):
            timecuda, stdcuda = run_benchmark(buildcuda, sizes, testname = testname, nruns = nruns, btype = 'cuda')
            Stimecuda, Sstdcuda = run_benchmark(buildcuda, Ssizes, nruns = nruns, precision = 32, testname = testname, btype = 'cuda')
            savez_dict['timecuda'] = timecuda
            savez_dict['Stimecuda'] = Stimecuda
            savez_dict['stdcuda'] = stdcuda
            savez_dict['Sstdcuda'] = Sstdcuda
        #endif
    #endif
    np.savez(path+testname + "_" + platform + "_nruns" + str(nruns) + ".npz", **savez_dict)
    return

def collect_nsys_results(testname, platform, nruns, output_file):
    nruns = int(nruns)

    sizes = array('d',[1000, 10000, 100000,1000000,5000000,10000000, 50000000,100000000]) #10,100,1000,10000,100000,
    Ssizes = array('d',[1000, 10000, 100000,1000000,5000000,10000000, 50000000,100000000,200000000]) #10,100,1000,10000,100000,

    val = list(sizes)
    sizes_gb = [v*8*4/1e9 for v in val] #(nb of doubles)*(bytes for doubles)*(4=dim of LVector)/(bytes in GB)
    sizes_gb = array('d', sizes_gb)

    val = list(Ssizes)
    Ssizes_gb = [v*4*4/1e9 for v in val] #(nb of floats)*(bytes for floats)*(4=dim of LVector)/(bytes in GB)
    Ssizes_gb = array('d', Ssizes_gb)

    platform = platform.lower()

    if ('buffers' in platform) or ('buf' in platform) or ('bf' in platform):
        memtype = 'BUF'
    elif ('device_pointers' in platform) or ('devptr' in platform) or ('dp' in platform):
        memtype = 'PTR'
    else:
        memtype = ''
    
    if (platform == 'cpu'):
        run_nsys_benchmark(buildacpp, sizes, memtype = memtype, testname = testname, nruns = nruns, btype = 'cpu', output_file = output_file)
        run_nsys_benchmark(buildacpp, Ssizes, memtype = memtype, precision = 32, testname = testname, nruns = nruns, btype = 'cpu', output_file = output_file)
    else:
        if ("cuda" in platform):
            run_nsys_benchmark(buildcuda, sizes, testname = testname, nruns = nruns, btype = 'cuda', output_file = output_file)
            run_nsys_benchmark(buildcuda, Ssizes, precision = 32,  nruns = nruns, testname = testname, btype = 'cuda', output_file = output_file)
        #endif

        run_nsys_benchmark(buildacpp, sizes, memtype = memtype, testname = testname, nruns = nruns, btype = 'sycl', output_file = output_file)
        run_nsys_benchmark(buildacpp, Ssizes, memtype = memtype, precision = 32,  testname = testname, nruns = nruns, btype = 'sycl', output_file = output_file)
        
        if (platform != "hip"):
            run_nsys_benchmark(buildoneapi, sizes, memtype = memtype, testname = testname, nruns = nruns, btype = 'sycl', output_file = output_file)
            run_nsys_benchmark(buildoneapi, Ssizes, memtype = memtype, precision = 32, testname = testname, nruns = nruns, btype = 'sycl', output_file = output_file)
        #endif

    #endif
    #np.savez(path+testname + "_" + platform + "_nruns" + str(nruns) + ".npz", **savez_dict)
    return

if __name__ == "__main__":
    #collect_results(sys.argv[1],sys.argv[2], sys.argv[3])
    if len(sys.argv) > 4:
        output_file = sys.argv[4]
    else:
        output_file = f"{time.strftime('%Y%m%d-%H%M%S')}"
    collect_nsys_results(sys.argv[1],sys.argv[2], sys.argv[3], output_file)

