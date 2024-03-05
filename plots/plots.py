from array import array
import subprocess
import math
import os
import sys
import time
import numpy as np
from pathlib import Path

nsys = "/usr/local/cuda-12.2/bin/nsys"

base_header = "platform,N,env,type,testname,precision"

def result2list(exe, N, nruns, env = None):
    print(exe, str(int(N)), str(int(nruns)))
    result = subprocess.run([exe, str(int(N)), str(int(nruns)) ], stdout=subprocess.PIPE, env = env).stdout.decode('utf-8')
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

def run_benchmark(builddir, sizes, 
                  nruns = 1,                         
                  testname = "InvariantMasses", 
                  precision = 64, 
                  btype = 'cpu', 
                  environment = 'cuda',
                  memtype = '',
                  platform = '',
                  output_file = 'output_file'):
    
    if precision == 32:
        testname = 'S'+testname

    env = None

    if btype == 'cpu':
        exe = os.path.join(builddir,'testx',testname)
        environment = 'CPU'
    elif btype == 'cuda':
        exe = os.path.join(builddir,'testx',testname+'CUDA')
        environment = 'CUDA'
        env = {'CUDA_VISIBLE_DEVICES':device_nb}
    elif btype == 'sycl':
        exe = os.path.join(builddir,'testx',testname+'SYCL')
        if "oneapi" in builddir.lower():
            environment = 'oneAPI'
            if environment == 'cuda':
                env = {'ONEAPI_DEVICE_SELECTOR': 'cuda:'+device_nb}
        if ("acpp" in builddir.lower()) or ("adaptivecpp" in builddir.lower()):
            environment = 'AdaptiveCPP'
            if environment == 'cuda':
                env = {'ACPP_VISIBILITY_MASK': 'cuda', 'CUDA_VISIBLE_DEVICES':device_nb}
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
    
    
    for i, N in enumerate(sizes):
        #run test
        time = result2list(exe,N,nruns, env = env)
        
        if (len(time)<=0):
            return timeopt, timestd
        print(time, '\n')
        
        # If output file does not exist, write headers
        if not Path(f"{output_file}/timing").exists():
            with open(f"{output_file}/timing", "w") as file_handler:
                header = "Time"
                file_handler.write(f"{base_header},{header}\n")
        # Write row
        out_base = f"{platform},{int(N)},{environment},{memtype},{testname},{precision}"
        with open(f"{output_file}/timing", "a") as file_handler:
            file_handler.write("\n".join([f"{out_base},{t}" for t in time]) )
            file_handler.write("\n")

        timeopt.append(np.amin(time))
        timestd.append(np.std(time)) 
    #endfor
    return timeopt, timestd

def run_nsys_benchmark(builddir, sizes, 
                       nruns = 1, 
                       testname = "InvariantMasses", 
                       environment = 'cuda',
                       memtype = '',
                       platform = '',
                       precision = 64,
                       btype = 'cpu', 
                       output_file = 'output_file'):

    if precision == 32:
        testname = 'S'+testname
    if btype == 'cpu':
        exe = os.path.join(builddir,'testx',testname)
        implemetation = 'CPU'
    elif btype == 'cuda':
        exe = os.path.join(builddir,'testx',testname+'CUDA')
        implemetation = 'CUDA'
        environ = {'CUDA_VISIBLE_DEVICES':device_nb}
    elif btype == 'sycl':
        exe = os.path.join(builddir,'testx',testname+'SYCL')
        if "oneapi" in builddir.lower():
            implemetation = 'oneAPI'
            environ = {'ONEAPI_DEVICE_SELECTOR':'cuda:'+device_nb}
        if ("acpp" in builddir.lower()) or ("adaptivecpp" in builddir.lower()):
            implemetation = 'AdaptiveCPP'
            environ = {'CUDA_VISIBLE_DEVICES':device_nb, 'ACPP_VISIBILITY_MASK':'cuda' }
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
                                    env=environ,
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
        out_base = f"{platform},{int(N)},{implemetation},{memtype},{testname},{precision}"
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

def collect_results(testname, platform, environment, memory_model, nruns, output_file):
    nruns = int(nruns)

    exponents_64 = [12, 15, 18, 21, 24, 26, 27]
    exponents_32 = [12, 15, 18, 21, 24, 27, 26, 28]

    if (platform.lower() == 'a100'):
        exponents_64.extend([28, 29])
        exponents_32.extend([28, 29, 30])
    #endif 
    if (platform.lower() == 'l4'):
        exponents_64.extend([28])
        exponents_32.extend([28, 29])    
    #endif 
    sizes = array('d',[2**i for i in exponents_64]) #10,100,1000,10000,100000,
    Ssizes = array('d',[2**i for i in exponents_32]) #10,100,1000,10000,100000,

    val = list(sizes)
    sizes_gb = [v*8*4/1e9 for v in val] #(nb of doubles)*(bytes for doubles)*(4=dim of LVector)/(bytes in GB)
    sizes_gb = array('d', sizes_gb)

    val = list(Ssizes)
    Ssizes_gb = [v*4*4/1e9 for v in val] #(nb of floats)*(bytes for floats)*(4=dim of LVector)/(bytes in GB)
    Ssizes_gb = array('d', Ssizes_gb)

    if  ('buf' in implementation) or ('bf' in implementation):
        memtype = 'BUF'
    elif ('device_pointers' in implementation) or ('dp' in implementation) or ('ptr' in implementation):
        memtype = 'PTR'
    else:
        memtype = ''

    savez_dict = {}
    savez_dict['sizes'] = sizes 
    savez_dict['Ssizes'] = Ssizes 
    savez_dict['sizes_gb'] = sizes_gb
    savez_dict['Ssizes_gb'] = Ssizes_gb 
    
    if (environment == 'cpu'):
        timecpu, stdcpu = run_benchmark(buildcpu, sizes, platform=platform, environment=environment, memtype = memtype, testname = testname, nruns = nrunsx, btype = 'cpu', output_file = output_file)
        Stimecpu, Sstdcpu = run_benchmark(buildcpu, sizes, platform=platform, environment=environment, precision = 32, memtype = memtype, testname = testname, nruns = nruns, btype = 'cpu', output_file = output_file)
        savez_dict['timecpu'] = timecpu
        savez_dict['Stimecpu'] = Stimecpu
        savez_dict['stdcpu'] = stdcpu
        savez_dict['Sstdcpu'] = Sstdcpu
    else:
        timeoneapi, stdoneapi = run_benchmark(buildoneapi, sizes, platform = platform, environment=environment, memtype = memtype, testname = testname, nruns = nruns, btype = 'sycl', output_file = output_file)
        Stimeoneapi, Sstdoneapi = run_benchmark(buildoneapi, sizes, platform = platform, environment=environment, precision =32, memtype = memtype, testname = testname, nruns = nruns, btype = 'sycl', output_file = output_file)
        savez_dict['timeoneapi'] = timeoneapi
        savez_dict['Stimeoneapi'] = Stimeoneapi
        savez_dict['stdoneapi'] = stdoneapi
        savez_dict['Sstdoneapi'] = Sstdoneapi

        timeacpp, stdacpp = run_benchmark(buildacpp, sizes, platform = platform, memtype = memtype, environment=environment, testname = testname, nruns = nruns, btype = 'sycl', output_file = output_file)
        Stimeacpp, Sstdacpp = run_benchmark(buildacpp, sizes, platform = platform, precision =32, environment=environment, memtype = memtype, testname = testname, nruns = nruns, btype = 'sycl', output_file = output_file)
        savez_dict['timeacpp'] = timeacpp
        savez_dict['Stimeacpp'] = Stimeacpp
        savez_dict['stdacpp'] = stdacpp
        savez_dict['Sstdacpp'] = Sstdacpp

        if ("cuda" in environment):
            timecuda, stdcuda = run_benchmark(buildcuda, sizes, platform = platform, memtype = memtype, environment=environment, testname = testname, nruns = nruns, btype = 'cuda', output_file = output_file)
            Stimecuda, Sstdcuda = run_benchmark(buildcuda, sizes, platform = platform, precision =32, environment=environment, memtype = memtype, testname = testname, nruns = nruns, btype = 'cuda', output_file = output_file)
            savez_dict['timecuda'] = timecuda
            savez_dict['Stimecuda'] = Stimecuda
            savez_dict['stdcuda'] = stdcuda
            savez_dict['Sstdcuda'] = Sstdcuda
        #endif
    #endif
    #np.savez(os.path.join(path, platform+testname + "_" + implementation + "_nruns" + str(nruns) + ".npz"), **savez_dict)
    return

def collect_nsys_results(testname, platform, environment, memtype, nruns, output_file):
    nruns = int(nruns)

    #sizes = array('d',[1000, 10000, 100000,1000000,5000000,10000000, 50000000,100000000]) #10,100,1000,10000,100000,
    #Ssizes = array('d',[1000, 10000, 100000,1000000,5000000,10000000, 50000000,100000000,200000000]) #10,100,1000,10000,100000,
    exponents_64 = [12, 15, 18, 21, 24, 26, 27]
    exponents_32 = [12, 15, 18, 21, 24, 27, 26, 28]

    if (platform.lower() == 'a100'):
        exponents_64.extend([28, 29])
        exponents_32.extend([28, 29, 30])
    #endif 
    if (platform.lower() == 'l4'):
        exponents_64.extend([28])
        exponents_32.extend([28, 29])    
    #endif 
    sizes = array('d',[2**i for i in exponents_64]) #10,100,1000,10000,100000,
    Ssizes = array('d',[2**i for i in exponents_32]) #10,100,1000,10000,100000,

    val = list(sizes)
    sizes_gb = [v*8*4/1e9 for v in val] #(nb of doubles)*(bytes for doubles)*(4=dim of LVector)/(bytes in GB)
    sizes_gb = array('d', sizes_gb)

    val = list(Ssizes)
    Ssizes_gb = [v*4*4/1e9 for v in val] #(nb of floats)*(bytes for floats)*(4=dim of LVector)/(bytes in GB)
    Ssizes_gb = array('d', Ssizes_gb)

    if (environment == 'cpu'):
        run_nsys_benchmark(buildcpu, sizes, platform = platform, environment = environment, memtype = memtype, testname = testname, nruns = nruns, btype = 'cpu', output_file = output_file)
        run_nsys_benchmark(buildcpu, Ssizes, platform = platform, environment = environment, memtype = memtype, precision = 32, testname = testname, nruns = nruns, btype = 'cpu', output_file = output_file)
    else:
        if (environment.lower() == "cuda"):
            run_nsys_benchmark(buildcuda, sizes, platform = platform, environment = environment, memtype = memtype, testname = testname, nruns = nruns, btype = 'cuda', output_file = output_file)
            run_nsys_benchmark(buildcuda, Ssizes, platform = platform, environment = environment, memtype = memtype, precision = 32,  nruns = nruns, testname = testname, btype = 'cuda', output_file = output_file)
        #endif

        run_nsys_benchmark(buildacpp, sizes, platform = platform, environment = environment, memtype = memtype, testname = testname, nruns = nruns, btype = 'sycl', output_file = output_file)
        run_nsys_benchmark(buildacpp, Ssizes, platform = platform, environment = environment, memtype = memtype, precision = 32,  testname = testname, nruns = nruns, btype = 'sycl', output_file = output_file)
        
        run_nsys_benchmark(buildoneapi, sizes, platform = platform, environment = environment, memtype = memtype, testname = testname, nruns = nruns, btype = 'sycl', output_file = output_file)
        run_nsys_benchmark(buildoneapi, Ssizes, platform = platform, environment = environment, memtype = memtype, precision = 32, testname = testname, nruns = nruns, btype = 'sycl', output_file = output_file)
    #endif
    return

if __name__ == "__main__":
    # Arguments
    # sys.argv[1] path of script
    # sys.argv[2] testname
    # sys.argv[3] platform (i.e. gpu model)
    # sys.argv[4] environment ("cuda" for nvidia gpus, "hip" for amd gpus, "cpu"...)
    # sys.argv[5] device index
    # sys.argv[6] sycl memory model ("buf" for buffers+accessors, "ptr" for device pointers)
    # sys.argv[7] number of test repetitions
    # sys.argv[8] output file path
    if len(sys.argv) > 8:
        output_file = sys.argv[8]
    else:
        output_file = f"{time.strftime('%Y%m%d-%H%M%S')}"
    #
    global path, buildcuda, buildoneapi, buildacpp, buildcpu, device_nb
    path = sys.argv[1]
    testname = sys.argv[2] 
    platform = sys.argv[3]
    environment = sys.argv[4]
    device_nb  = sys.argv[5]
    memory_model = sys.argv[6]
    implementation = environment+'_'+memory_model 
    nruns = sys.argv[7]
    #
    buildcpu   = os.path.join(path, '../build_cpu') 
    buildcuda   = os.path.join(path, '../build_cuda')
    buildoneapi = os.path.join(path, '../build_oneapi_'+memory_model.lower())
    buildacpp   = os.path.join(path,'../build_acpp_'+memory_model.lower())
    #
    print(testname, platform, environment, memory_model, nruns, output_file)
    collect_results(testname, platform, environment, memory_model, nruns, output_file)
    if (environment.lower() == 'cuda'):
        collect_nsys_results(testname, platform, environment, memory_model, nruns, output_file)
    
        