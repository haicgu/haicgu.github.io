Notes from installing AI software stack
=======================================


NPU driver
----------

The npu driver has to be patched and repacked. There are following problems:

Only A800 3000:

* Red Hat kernel api not respected (Also API in v. 5.7)

  * pci_cleanup_aer_uncorrect_error_status -> pci_aer_clear_nonfatal_status
* ndo timeout something something two arguments instead of one (TODO: look up what it was again)

Both A800 3000 and 9000:

* Rocky is not recognized which leads to a wrong dkms path for the check if the module exists

  * Add Rocky to a couple lines which check lsb_release output (TODO: which file?)
* install script checks for local users/groups only.

  * /etc/{passwd,group} checks -> getent checks
* TODO: provide patch that fixes all problems

CANN CE dl link: https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/5.0.2.alpha005/Ascend-cann-toolkit_5.0.2.alpha005_linux-aarch64.run


The non-local user fix works on a running node, but not in a chroot. Workaround::

  getent passwd hiaiowner >> $CHROOT/etc/passwd
  getent group hiai >> $CHROOT/etc/group
  # enter chroot
  # run the installer
  # exit chroot
  # Delete the lines in $CHROOT/etc/passwd and $CHROOT/etc/group after installation
  # DO NOT UNDER ANY CIRCUMSTANCES MESS WITH /etc/passwd OR /etc/group ON A RUNNING SYSTEM!

The ``/etc/rc.d/init.d/host_sys_init.sh`` file needs to be modified to wait until network users are available (Figure out a good way to do this, maybe wait for sssd or force-join-ipa services? Or wait until id $username works (try 10 times in 30 sec interval?))

* Current workaround - test with getent in a loop for 10 min
* This still fails sometimes
* Manually do
  ``sudo systemctl restart host_sys_init``
  when username and group available

CANN Toolkit
------------

* CANN Installers mess around too much with users/groups/permissions, again requiring patching
* Amount of effort to fix this manually too high, writing script
* Actually directly integrating the fixes into an EasyBlock/EasyConfig is better
* EasyBlock is `here`__
* EasyConfig is `here`__
* Essentially what I'm doing:

  * unpacking the installer
  * unpacking the subinstallers
  * looking for scripts in the extracted subinstallers
  * looking for offending functions inside the scripts 
  * substituting inside of them with fixes.
  * Applying a patch on top that fixes some things that were difficult to do with substitutions
  * running the subinstaller scripts manually with what is basically reverse-engineered parameters
* Installers are kind of inconsistent and I'm not sure how to repack them, so this all feels more like a hack

__ https://gitlab.jsc.fz-juelich.de/nassyr1/juawei-easyconfigs/-/blob/master/Custom_EasyBlocks/2021a/cann_package.py
__ https://gitlab.jsc.fz-juelich.de/nassyr1/juawei-easyconfigs/-/blob/master/Golden_Repo/2021a/c/CANN-Toolkit/CANN-Toolkit-5.0.2.alpha005-goolf-2021a.9-Python-3.7.5.eb


AscendCL example
~~~~~~~~~~~~~~~~

AscendCL seems to be working well. I wrote a rather verbose single-file C++ example trying to understand how it works.

Source of the example:

.. code-block:: c++

    [snassyr@dev GEMM]$ cat gemm.cpp
    #include <algorithm>
    #include <cmath>
    #include <chrono>
    #include <cstdint>
    #include <cstdlib>
    #include <filesystem>
    #include <random>
    #include <string>
    #include <thread>
    #include <vector>

    #include <acl/acl.h>
    #include <acl/ops/acl_cblas.h>
    #include <fmt/core.h>
    #include <fmt/os.h>
    #include <fmt/color.h>
    #include <cblas.h>

    enum class loglevel : std::uint32_t
    {
        error=1,
        warning,
        info,
        success,
        debug,
    };

    loglevel threshold_ll = loglevel::error;

    template<typename S, typename ...Args>
    void log(loglevel level, const S& format_str, const Args&... args)
    {
        if(threshold_ll < level)
        {
            return;
        }
        std::string level_str{"INFO"};
        fmt::text_style style;
        switch(level)
        {
            case loglevel::debug:
                level_str = "DEBUG";
                break;
            case loglevel::success:
                style = fmt::fg(fmt::color::green) | fmt::emphasis::bold;
                level_str = "SUCCESS";
                break;
            case loglevel::info:
                level_str = "INFO";
                break;
            case loglevel::warning:
                style = fmt::fg(fmt::color::yellow) | fmt::emphasis::bold;
                level_str = "WARNING";
                break;
            case loglevel::error:
                style = fmt::fg(fmt::color::red) | fmt::emphasis::bold;
                level_str = "ERROR";
                break;
            default:
                break;
        }
        level_str = fmt::format(style, "[{:^10}]",level_str);
        fmt::print("{}{}\n", level_str, fmt::format(format_str, args...));
    }

    template<typename S, typename ...Args>
    void log(loglevel level, std::size_t thread_id, const S& format_str, const Args&... args)
    {
        log(level, fmt::format("[TID:{:>3}] {}", thread_id, format_str), args...);
    }



    void worker_function(std::uint32_t M, std::uint32_t N, std::uint32_t K, std::size_t id, std::string op_model_dir)
    {
        typedef loglevel ll;
        log(ll::debug, id, "Starting worker");

        // not sure if auto&...  necessary for some cases
        auto acl_try = [id](auto function, auto desc, auto... args)
        {
            log(ll::debug, id, fmt::format("Attempting to {}",desc));
            auto err_code = function(args...);
            if(ACL_SUCCESS != err_code)
            {
                log(ll::error, id, fmt::format("Failed to {}. Error Code: {}", desc, err_code));
                return false;
            }
            log(ll::success, id, fmt::format("Succeeded to {}", desc));
            return true;
        };


        if (!acl_try(aclrtSetDevice, fmt::format("set compute device to {}",id), id)) return;

        aclrtContext context;
        if (!acl_try(aclrtCreateContext, "create context", &context, id)) return;

        if (!acl_try(aclrtSetCurrentContext, fmt::format("set current context to {}", context), context)) return;

        aclrtStream stream;
        if (!acl_try(aclrtCreateStream, "create stream", &stream)) return;

        aclrtRunMode mode;
        if (!acl_try(aclrtGetRunMode, "get run mode", &mode)) return;

        std::string rm_str;

        if (ACL_HOST == mode)
        {
            rm_str ="ACL_HOST";
        }
        else if (ACL_DEVICE == mode)
        {
            rm_str = "ACL_DEVICE";
        }
        else
        {
            rm_str = "unknown";
        }
        log(ll::debug, id, "run mode is {}", rm_str);


        // Reference data
        std::vector<float> a(M*K);
        std::vector<float> b(K*N);
        std::vector<float> c(M*N);
        float              alpha;
        float              beta;
        std::vector<float> c_ref(M*N);

        // Seed a mersenne twister with "real" random data
        std::random_device source;
        std::size_t data_size = std::mt19937_64::state_size*sizeof(std::mt19937_64::result_type);
        std::vector<std::random_device::result_type> rnd_data((data_size-1)/sizeof(source())+1);
        std::generate(std::begin(rnd_data), std::end(rnd_data), std::ref(source));
        std::seed_seq seeds(std::begin(rnd_data), std::end(rnd_data));
        std::mt19937_64 gen(seeds);
        std::uniform_real_distribution<float> dist(0.0f,2.0f);

        // generate random data
        log(ll::info, id, "Generating A data");
        std::generate_n(std::begin(a),M*K,[&](){return dist(gen);});
        log(ll::info, id, "Generating B data");
        std::generate_n(std::begin(b),K*N,[&](){return dist(gen);});
        log(ll::info, id, "Generating C data");
        std::generate_n(std::begin(c),M*N,[&](){return dist(gen);});
        alpha = dist(gen);
        beta  = dist(gen);


        bool mem_error = false;


        // Device pointers
        void* device_a = nullptr,
            * device_b = nullptr,
            * device_c = nullptr,
            * device_alpha = nullptr,
            * device_beta = nullptr;
        
        auto allocate = [acl_try,id,&mem_error](void** ptr, auto size, auto desc)
        {
            if (!acl_try(aclrtMalloc,
                         fmt::format("allocate device memory for {}", desc),
                         ptr, size, ACL_MEM_MALLOC_NORMAL_ONLY))
            {
                mem_error = true;
                return;
            }
        };

        allocate(&device_a, M*K*aclDataTypeSize(ACL_FLOAT16), "matrix A");
        if(mem_error) return;
        allocate(&device_b, K*N*aclDataTypeSize(ACL_FLOAT16), "matrix B");
        if(mem_error) return;
        allocate(&device_c, M*N*aclDataTypeSize(ACL_FLOAT16), "matrix C");
        if(mem_error) return;
        allocate(&device_alpha, aclDataTypeSize(ACL_FLOAT16), "alpha");
        if(mem_error) return;
        allocate(&device_beta,  aclDataTypeSize(ACL_FLOAT16), "beta");
        if(mem_error) return;

        // Host pointers
        void* host_a = nullptr,
            * host_b = nullptr,
            * host_c = nullptr,
            * host_alpha = nullptr,
            * host_beta = nullptr;

        if(ACL_HOST == mode)
        {
            auto allocate = [acl_try,id,&mem_error](void** ptr, auto size, auto desc)
            {
                if (!acl_try(aclrtMallocHost,
                             fmt::format("allocate host memory for {}", desc),
                             ptr, size))
                {
                    mem_error = true;
                    return;
                }
            };

            allocate(&host_a, M*K*aclDataTypeSize(ACL_FLOAT16), "matrix A");
            if(mem_error) return;
            allocate(&host_b, K*N*aclDataTypeSize(ACL_FLOAT16), "matrix B");
            if(mem_error) return;
            allocate(&host_c, M*N*aclDataTypeSize(ACL_FLOAT16), "matrix C");
            if(mem_error) return;
            allocate(&host_alpha, aclDataTypeSize(ACL_FLOAT16), "alpha");
            if(mem_error) return;
            allocate(&host_beta,  aclDataTypeSize(ACL_FLOAT16), "beta");
            if(mem_error) return;
        }
        else
        {
            host_a = device_a;
            host_b = device_b;
            host_c = device_c;
            host_alpha = device_alpha;
            host_beta  = device_beta;
        }


        // fill host memory objects with data
        auto convert = [id,&mem_error](void* ptr, const auto& data, std::size_t count, auto desc)
        {
            log(ll::debug, id, "Converting {} from float to aclFloat16", desc);
            std::transform(std::begin(data), std::end(data), (aclFloat16*) ptr, [](float val)->aclFloat16
                    {
                        return aclFloatToFloat16(val);
                    });
            log(ll::success, id, "Converted {} from float to aclFloat16", desc);
        };

        log(ll::info, id, "converting/moving matrices to ACL_FLOAT16 in host memory");
        convert(host_a, a, M*K, "matrix A");
        if(mem_error) return;
        convert(host_b, b, K*N, "matrix B");
        if(mem_error) return;
        convert(host_c, c, M*N, "matrix C");
        if(mem_error) return;
        convert(host_alpha, std::array{alpha}, 1, "alpha");
        if(mem_error) return;
        convert(host_beta, std::array{beta},  1, "beta");
        if(mem_error) return;

        // copy data to device if separate memory
        if(ACL_HOST == mode)
        {
            auto copy_to_device = [acl_try,id,&mem_error,stream] (void* src, void* dest, std::size_t size, auto desc)
            {
                if (!acl_try(aclrtMemcpyAsync,
                             fmt::format("start async copy of {} from host to device", desc),
                             dest, size, src, size,
                             ACL_MEMCPY_HOST_TO_DEVICE, stream))
                {
                    mem_error = true;
                    return;
                }
            };

            log(ll::info, id, "moving matrices to device memory");
            copy_to_device(host_a, device_a, M*K*aclDataTypeSize(ACL_FLOAT16), "matrix A");
            if(mem_error) return;
            copy_to_device(host_b, device_b, K*N*aclDataTypeSize(ACL_FLOAT16), "matrix B");
            if(mem_error) return;
            copy_to_device(host_c, device_c, M*N*aclDataTypeSize(ACL_FLOAT16), "matrix C");
            if(mem_error) return;
            copy_to_device(host_alpha, device_alpha, aclDataTypeSize(ACL_FLOAT16), "alpha");
            if(mem_error) return;
            copy_to_device(host_beta,  device_beta, aclDataTypeSize(ACL_FLOAT16), "beta");
            if(mem_error) return;


            if (!acl_try(aclrtSynchronizeStream, "synchronize stream to finish copies", stream)) return;
        }
        
        // measure total runtime
        
        using hrc=std::chrono::high_resolution_clock;
        using std::chrono::duration_cast;
        using us=std::chrono::microseconds;

        auto start = hrc::now();
        // Run gemm
        log(ll::info, id, "running GEMM");
        if (!acl_try(aclblasGemmEx, "start async GEMM on device",
                     ACL_TRANS_N, ACL_TRANS_N, ACL_TRANS_N,
                     M, N, K,
                     device_alpha, device_a, -1, ACL_FLOAT16,
                     device_b, -1, ACL_FLOAT16,
                     device_beta, device_c, -1, ACL_FLOAT16,
                     ACL_COMPUTE_HIGH_PRECISION,
                     stream))
        {
                return;
        }
        
        if (!acl_try(aclrtSynchronizeStream, "synchronize stream to finish GEMM", stream)) return;

        auto gemmend = hrc::now();

        // Copy the result if separate memory
        if(ACL_HOST == mode)
        {
            std::uint32_t size = M*N*aclDataTypeSize(ACL_FLOAT16);

            if (!acl_try(aclrtMemcpy, "copy result from device to host",
                         host_c, size, device_c, size,
                              ACL_MEMCPY_DEVICE_TO_HOST))
            {
                mem_error = true;
                return;
            }

        }
        auto copyend = hrc::now();

        fmt::print("Thread {}: Gemm finished in {} us (including startup latency), result copy finished in {} us\n",
                id,
                duration_cast<us>(gemmend-start).count(),
                duration_cast<us>(copyend-gemmend).count()
                );

        log(ll::info, id, "running reference fp32 SGEMM");
        // Calculate reference
        cblas_sgemm(CblasRowMajor,
                CblasNoTrans,CblasNoTrans,
                M, N, K, 
                alpha, a.data(), K,
                b.data(), N,
                beta, c.data(), N);

        // compare results
        log(ll::info, id, "comparing results");
        float threshold = 0.01; // percent/100 max difference
        bool value_error = false;
        for (std::size_t i = 0; i < M; i++)
        {
            for (std::size_t j = 0; j < N; j++)
            {
                auto diff = std::abs((c[i*N+j] - aclFloat16ToFloat(((aclFloat16*)host_c)[i*N+j]))/c[i*N+j]);
                if (diff > threshold)
                {
                    log(ll::warning, id, "Calculated result at ({},{}) differs from reference by {}% (> {} %)",
                            i, j, diff*100.0f, threshold*100.0f);
                    log(ll::warning, id, "result({0},{1}) = {2}, reference({0},{1}) = {3})",
                            i, j,
                            c[i*N+j], aclFloat16ToFloat(((aclFloat16*)host_c)[i*N+j]));
                    value_error = true;
                }
            }
        }

        if(value_error)
        {
            fmt::print("Thread {}: At least 1 value was off by more than {}%\n", id, threshold*100.0f);
        }
        else
        {
            fmt::print("Thread {}: All values within {}% from reference\n", id, threshold*100.0f);
        }


        // Free memory
        if(ACL_HOST == mode)
        {
            auto deallocate = [acl_try,id,&mem_error](void* ptr, auto desc)
            {
                if (!acl_try(aclrtFreeHost, fmt::format("free host memory for {}", desc),ptr))
                {
                    mem_error = true;
                    return;
                }
            };
            deallocate(host_a, "matrix A");
            if(mem_error) return;
            deallocate(host_b, "matrix B");
            if(mem_error) return;
            deallocate(host_c, "matrix C");
            if(mem_error) return;
            deallocate(host_alpha, "alpha");
            if(mem_error) return;
            deallocate(host_beta,  "beta");
            if(mem_error) return;
        }

        auto deallocate = [acl_try,id,&mem_error](void* ptr, auto desc)
        {
            if (!acl_try(aclrtFree, fmt::format("free device memory for {}", desc),ptr))
            {
                mem_error = true;
                return;
            }
        };
        deallocate(device_a, "matrix A");
        if(mem_error) return;
        deallocate(device_b, "matrix B");
        if(mem_error) return;
        deallocate(device_c, "matrix C");
        if(mem_error) return;
        deallocate(device_alpha, "alpha");
        if(mem_error) return;
        deallocate(device_beta,  "beta");
        if(mem_error) return;




        // Cleanup from here on
        if(!acl_try(aclrtDestroyStream, "destroy stream", stream)) return;
        if(!acl_try(aclrtDestroyContext, "destroy context", context)) return;
        if(!acl_try(aclrtResetDevice, fmt::format("Reset device {}",id),id)) return;

        log(ll::debug, id, "Shutting down worker");
    }

    int main(int argc, char* argv[])
    {
        namespace fs=std::filesystem;
        typedef loglevel ll;
        if(argc != 5)
        {
            fmt::print(
                "Usage: {} soc_version M N K\n\n"
                "Where M, N and K are matrix dimensions and\n"
                "soc_version is the Ascend SoC to target (Ascend910, Ascend310, ...)\n"
                "This will use each detected Ascend device to calculate C=alpha*AB+beta*C\n"
                "where A is a MxK and B is a KxN fp16 matrix filled with randomly generated\n"
                "numbers and alpha, beta are fp16 random scalars. You can controll the\n",
                "verbosity by setting the environment variable LOGLEVEL={1..5} (default: 1)\n",
                argv[0]);
            return -1;
        }

        std::string soc_version(argv[1]);
        std::uint32_t M = std::atol(argv[2]);
        std::uint32_t N = std::atol(argv[3]);
        std::uint32_t K = std::atol(argv[4]);


        const char* llenv = std::getenv("LOGLEVEL");
        if(nullptr != llenv)
        {
            threshold_ll = static_cast<ll>(std::atol(llenv));
        }

        // consider deduplicating code (threaded version in worker_function)
        auto fun_try = [](const auto success_code, const auto function, const auto& desc, auto... args)
        {
            log(ll::debug, fmt::format("Attempting to {}",desc));
            auto err_code = function(args...);
            if(success_code != err_code)
            {
                log(ll::error, fmt::format("Failed to {}. Error Code: {}", desc, err_code));
                return false;
            }
            log(ll::success, fmt::format("Succeeded to {}", desc));
            return true;
        };

        auto acl_try = [fun_try](const auto function, const auto& desc, auto... args)
        {
            return fun_try(ACL_SUCCESS, function, desc, args...);
        };

        fmt::print("Matrix dimensions: M={}, N={}, K={}\n", M, N, K);

        // Create the operator

        // check if atc binary exists
        if (!fun_try(0, std::system, "confirm if \"atc --version\" works/atc in $PATH", "atc --version > /dev/null")) return -1;

        // Just to make sure also check for ccec (it's used internally by atc)
        if (!fun_try(0, std::system, "confirm if \"ccec --version\" works/ccec in $PATH", "ccec --version > /dev/null")) return -1;


        auto op_model_dir = "op_model";
        if(!fs::exists(op_model_dir))
        {
            fs::create_directory(op_model_dir);
        }
        // Create gemm json
        auto matdesc = [](auto&& fmtvar, auto dim1, auto dim2)
        {
            fmtvar.print(
                "{{\n"
                "    \"format\": \"ND\",\n"
                "    \"shape\" : [{},{}],\n"
                "    \"type\" : \"float16\"\n"
                "}}",
                dim1, dim2);
        };
        auto scaldesc = [](auto&& fmtvar)
        {
            fmtvar.print(
                "{{\n"
                "    \"format\": \"ND\",\n"
                "    \"shape\" : [],\n"
                "    \"type\" : \"float16\"\n"
                "}}");
        };
        auto attrdesc = [](auto&& fmtvar, auto name)
        {
            fmtvar.print(
                "{{\n"
                "    \"name\": \"{}\",\n"
                "    \"type\" : \"bool\",\n"
                "    \"value\" : false\n"
                "}}",
                name);
        };

        auto gemm_config = fmt::format("gemm_{}x{}x{}_{}",M,N,K,soc_version);
        auto gemm_js_path = fs::path(op_model_dir) / fmt::format("{}.json",gemm_config);
        if (fs::exists(gemm_js_path))
        {
            log(ll::info,"{} already exists, skipping operand description generation",gemm_js_path.string());
        }
        else
        {
            log(ll::info, "Creating operand description in {}", gemm_js_path.string());

            auto gemm_js = fmt::output_file(gemm_js_path.string());
            gemm_js.print(
                    "[\n"
                    "{{\n"
                    "\"op\" : \"GEMM\",\n"
                    "\"input_desc\" : [\n"
                    );

            matdesc(gemm_js, M, K);
            gemm_js.print(",\n");
            matdesc(gemm_js, K, N);
            gemm_js.print(",\n");
            matdesc(gemm_js, M, N);
            gemm_js.print(",\n");
            scaldesc(gemm_js);
            gemm_js.print(",\n");
            scaldesc(gemm_js);

            gemm_js.print(
                    "],\n"
                    );

            gemm_js.print(
                    "\"output_desc\" : [\n"
                    );

            matdesc(gemm_js, M, N);

            gemm_js.print(
                    "],\n"
                    );

            gemm_js.print(
                    "\"attr\" : [\n"
                    );

            attrdesc(gemm_js, "transpose_a");
            gemm_js.print(",\n");
            attrdesc(gemm_js, "transpose_b");

            gemm_js.print(
                    "]\n"
                    );



            gemm_js.print(
                    "}}\n"
                    "]"
                    );
        }

        // Compile gemm json to offline model
        auto gemm_om_path = fs::path(op_model_dir) / gemm_config;
        if (fs::exists(gemm_om_path))
        {
            log(ll::info,"{} already exists, skipping running atc",gemm_om_path.string());
        }
        else
        {
            auto atc_command = fmt::format("atc --singleop={} --output={} --soc_version={}",
                    gemm_js_path.string(), gemm_om_path.string(), soc_version);

            if (!fun_try(0, std::system, fmt::format("run command: {}", atc_command), atc_command.c_str())) return -1;
        }


        // Initialize AscendCL API
        if (!acl_try(aclInit, "initialize ACL", nullptr)) return -1;
        // Setting operand model directory
        auto omdir_str = gemm_om_path.string();
        if (!acl_try(aclopSetModelDir, fmt::format("set op model dir to {}", omdir_str), omdir_str.c_str())) return -1;

        // Query devices
        std::uint32_t device_count = 0;

        if (!acl_try(aclrtGetDeviceCount, "get device count", &device_count)) return -1;

        log(ll::info, "Number of usable devices: {}", device_count);
        if (0 >= device_count)
        {
            log(ll::error,"No usable devices found. aborting");
            return -1;
        }

        // Create a thread for each device
        std::vector<std::thread> device_threads;
        for(std::uint32_t i = 0; i < device_count; i++)
        {
            device_threads.push_back(std::thread(worker_function, M, N, K, i, op_model_dir));
        }

        for(auto& th : device_threads)
        {
            th.join();
        }
        
        // Deinit ACL API
        if (!acl_try(aclFinalize, "finalize ACL")) return -1;


        return 0;
    }


Building the example::

    [snassyr@dev ~]$ cd SourceCode/test/CANN/AscendCL/GEMM/
    [snassyr@dev GEMM]$ module load GCC/9.3.0 fmt OpenMPI CANN-Toolkit
    [snassyr@dev GEMM]$ g++ -std=c++17 -O2 -march=native -mcpu=native -o gemm gemm.cpp -pthread -lfmt -lopenblas -lascendcl -lacl_cblas


Running the example on a800-9000::

    [snassyr@dev GEMM]$ srun -p a800-9000 ./gemm Ascend910 256 256 256
    ATC start working now, please wait for a moment.
    ATC run success, welcome to the next use.

    Matrix dimensions: M=256, N=256, K=256
    Thread 0: Gemm finished in 3060 us (including startup latency), result copy finished in 24 us
    Thread 0: All values within 1% from reference
    Thread 1: Gemm finished in 2611 us (including startup latency), result copy finished in 26 us
    Thread 1: All values within 1% from reference
    Thread 2: Gemm finished in 2278 us (including startup latency), result copy finished in 32 us
    Thread 2: All values within 1% from reference
    Thread 3: Gemm finished in 2618 us (including startup latency), result copy finished in 32 us
    Thread 3: All values within 1% from reference
    Thread 4: Gemm finished in 2285 us (including startup latency), result copy finished in 32 us
    Thread 4: All values within 1% from reference
    Thread 5: Gemm finished in 2322 us (including startup latency), result copy finished in 23 us
    Thread 5: All values within 1% from reference
    Thread 6: Gemm finished in 2345 us (including startup latency), result copy finished in 24 us
    Thread 6: All values within 1% from reference
    Thread 7: Gemm finished in 2599 us (including startup latency), result copy finished in 32 us
    Thread 7: All values within 1% from reference

Running the example on a800-3000::

    [snassyr@dev GEMM]$ srun -p a800-3000 ./gemm Ascend310 256 256 256
    ATC start working now, please wait for a moment.
    ATC run success, welcome to the next use.

    Matrix dimensions: M=256, N=256, K=256
    Thread 0: Gemm finished in 3980 us (including startup latency), result copy finished in 107 us
    Thread 0: All values within 1% from reference
    Thread 1: Gemm finished in 3166 us (including startup latency), result copy finished in 98 us
    Thread 1: All values within 1% from reference
    Thread 2: Gemm finished in 3993 us (including startup latency), result copy finished in 98 us
    Thread 2: All values within 1% from reference
    Thread 3: Gemm finished in 3468 us (including startup latency), result copy finished in 99 us
    Thread 3: All values within 1% from reference
    Thread 4: Gemm finished in 3356 us (including startup latency), result copy finished in 96 us
    Thread 4: All values within 1% from reference
    Thread 5: Gemm finished in 3268 us (including startup latency), result copy finished in 97 us
    Thread 5: All values within 1% from reference
    Thread 6: Gemm finished in 3182 us (including startup latency), result copy finished in 98 us
    Thread 6: All values within 1% from reference
    Thread 7: Gemm finished in 4839 us (including startup latency), result copy finished in 96 us
    Thread 7: All values within 1% from reference
    Thread 8: Gemm finished in 3685 us (including startup latency), result copy finished in 97 us
    Thread 8: All values within 1% from reference
    Thread 9: Gemm finished in 3149 us (including startup latency), result copy finished in 96 us
    Thread 9: All values within 1% from reference
    Thread 10: Gemm finished in 3189 us (including startup latency), result copy finished in 95 us
    Thread 10: All values within 1% from reference
    Thread 11: Gemm finished in 3487 us (including startup latency), result copy finished in 100 us
    Thread 11: All values within 1% from reference
    Thread 12: Gemm finished in 3214 us (including startup latency), result copy finished in 104 us
    Thread 12: All values within 1% from reference
    Thread 13: Gemm finished in 3369 us (including startup latency), result copy finished in 99 us
    Thread 13: All values within 1% from reference
    Thread 14: Gemm finished in 3161 us (including startup latency), result copy finished in 96 us
    Thread 14: All values within 1% from reference
    Thread 15: Gemm finished in 3356 us (including startup latency), result copy finished in 97 us
    Thread 15: All values within 1% from reference
    Thread 16: Gemm finished in 3181 us (including startup latency), result copy finished in 99 us
    Thread 16: All values within 1% from reference
    Thread 17: Gemm finished in 3403 us (including startup latency), result copy finished in 97 us
    Thread 17: All values within 1% from reference
    Thread 18: Gemm finished in 3273 us (including startup latency), result copy finished in 95 us
    Thread 18: All values within 1% from reference
    Thread 19: Gemm finished in 115104 us (including startup latency), result copy finished in 31465 us
    Thread 19: All values within 1% from reference

CANN tfplugin
-------------

* I've tried installing `from the binary`__

  * This installs but importing the module results in unresolved symbols. Looking into it reveals that the tfplugin calls functions that are not defined in TensorFlow. Maybe they rely on an internal proprietary version of TensorFlow?
* I've tried installing `from the source`__

  * With a lot of patches, this installs, compiles and the module can be imported, but I never got any sample to actually run
  * I've tried multiple versions from 1.3.0 to 1.7.0, never successful

__ https://gitlab.jsc.fz-juelich.de/nassyr1/juawei-easyconfigs/-/blob/master/Golden_Repo/2021a/c/CANN-tfplugin/CANN-tfplugin-5.0.2.alpha005-goolf-2021a.9-Python-3.7.5.eb
__ https://gitlab.jsc.fz-juelich.de/nassyr1/juawei-easyconfigs/-/blob/master/Golden_Repo/2021a/c/CANN-tfplugin/CANN-tfplugin-1.7.0-goolf-2021a.9-Python-3.7.5.eb


Additional notes when following the installation manual
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Configuring NPU devices as described in https://support.huawei.com/enterprise/en/doc/EDOC1100206655/7643dda6/installing-software-packages-on-an-ascend-device

logged onto ml01-ipmi::

    ssh -o PreferredAuthentications=password -o PubkeyAuthentication=no Administrator@ml01-ipmi

get npuworkmode::

    iBMC:/->ipmcget -d npuworkmode
    ------------------ NPU BOARD WORK MODE ------------------
    NPU BOARD ID                  : 1
    WORK MODE                     : SMP
    ------------------ NPU BOARD WORK MODE ------------------
    NPU BOARD ID                  : 2
    WORK MODE                     : SMP


SMP means we have to configure the IPs with hccn_tool

Configuring in pairs on the same network: 0+4,1+5,2+6,3+7::

    hccn_tool -i 0 -ip -s address 10.45.100.1 netmask 255.255.255.0
    hccn_tool -i 4 -ip -s address 10.45.100.2 netmask 255.255.255.0

    hccn_tool -i 1 -ip -s address 10.45.101.1 netmask 255.255.255.0
    hccn_tool -i 5 -ip -s address 10.45.101.2 netmask 255.255.255.0

    hccn_tool -i 2 -ip -s address 10.45.102.1 netmask 255.255.255.0
    hccn_tool -i 6 -ip -s address 10.45.102.2 netmask 255.255.255.0

    hccn_tool -i 3 -ip -s address 10.45.103.1 netmask 255.255.255.0
    hccn_tool -i 7 -ip -s address 10.45.103.2 netmask 255.255.255.0

on ml02 there are 5 Atlas 300T cards, give each an ip on their own network::

    [snassyr@ml02 ~]$ npu-smi info
    +------------------------------------------------------------------------------+
    | npu-smi 20.2.2                       Version: 20.2.2                         |
    +-------------------+-----------------+----------------------------------------+
    | NPU     Name      | Health          | Power(W)          Temp(C)              |
    | Chip    Device    | Bus-Id          | AICore(%)         Memory-Usage(MB)     |
    +===================+=================+========================================+
    | 2048    310       | OK              | 12.8              46                   |
    | 0       0         | 0000:81:00.0    | 0                 2621 / 8192          |
    +-------------------+-----------------+----------------------------------------+
    | 2048    310       | OK              | 12.8              46                   |
    | 1       1         | 0000:82:00.0    | 0                 2621 / 8192          |
    +-------------------+-----------------+----------------------------------------+
    | 2048    310       | OK              | 12.8              47                   |
    | 2       2         | 0000:83:00.0    | 0                 2621 / 8192          |
    +-------------------+-----------------+----------------------------------------+
    | 2048    310       | OK              | 12.8              45                   |
    | 3       3         | 0000:84:00.0    | 0                 2621 / 8192          |
    +===================+=================+========================================+
    | 2049    310       | OK              | 12.8              49                   |
    | 0       4         | 0000:85:00.0    | 0                 2621 / 8192          |
    +-------------------+-----------------+----------------------------------------+
    | 2049    310       | OK              | 12.8              50                   |
    | 1       5         | 0000:86:00.0    | 0                 2621 / 8192          |
    +-------------------+-----------------+----------------------------------------+
    | 2049    310       | OK              | 12.8              50                   |
    | 2       6         | 0000:87:00.0    | 0                 2621 / 8192          |
    +-------------------+-----------------+----------------------------------------+
    | 2049    310       | OK              | 12.8              49                   |
    | 3       7         | 0000:88:00.0    | 0                 2621 / 8192          |
    +===================+=================+========================================+
    | 2050    310       | OK              | 12.8              49                   |
    | 0       8         | 0000:89:00.0    | 0                 2621 / 8192          |
    +-------------------+-----------------+----------------------------------------+
    | 2050    310       | OK              | 12.8              50                   |
    | 1       9         | 0000:8A:00.0    | 0                 2621 / 8192          |
    +-------------------+-----------------+----------------------------------------+
    | 2050    310       | OK              | 12.8              49                   |
    | 2       10        | 0000:8B:00.0    | 0                 2621 / 8192          |
    +-------------------+-----------------+----------------------------------------+
    | 2050    310       | OK              | 12.8              50                   |
    | 3       11        | 0000:8C:00.0    | 0                 2621 / 8192          |
    +===================+=================+========================================+
    | 2051    310       | OK              | 12.8              51                   |
    | 0       12        | 0000:8D:00.0    | 0                 2621 / 8192          |
    +-------------------+-----------------+----------------------------------------+
    | 2051    310       | OK              | 12.8              50                   |
    | 1       13        | 0000:8E:00.0    | 0                 2621 / 8192          |
    +-------------------+-----------------+----------------------------------------+
    | 2051    310       | OK              | 12.8              48                   |
    | 2       14        | 0000:8F:00.0    | 0                 2621 / 8192          |
    +-------------------+-----------------+----------------------------------------+
    | 2051    310       | OK              | 12.8              48                   |
    | 3       15        | 0000:90:00.0    | 0                 2621 / 8192          |
    +===================+=================+========================================+
    | 2052    310       | OK              | 12.8              53                   |
    | 0       16        | 0000:91:00.0    | 0                 2621 / 8192          |
    +-------------------+-----------------+----------------------------------------+
    | 2052    310       | OK              | 12.8              54                   |
    | 1       17        | 0000:92:00.0    | 0                 2621 / 8192          |
    +-------------------+-----------------+----------------------------------------+
    | 2052    310       | OK              | 12.8              56                   |
    | 2       18        | 0000:93:00.0    | 0                 2621 / 8192          |
    +-------------------+-----------------+----------------------------------------+
    | 2052    310       | OK              | 12.8              52                   |
    | 3       19        | 0000:94:00.0    | 0                 2621 / 8192          |
    +===================+=================+========================================+

The IDs are 2048,2049,2050,2051,2052

For some reason the driver doesn't install hccn_tool, so we copy it over from ml01 along with libdrvdsmi_host.so and libascend_hal.so. then as root::

    [root@ml02 snassyr]# LD_PRELOAD=./libdrvdsmi_host.so:./libascend_hal.so ./hccn_tool -i 2048 -ip -s address 10.45.0.1 netmask 255.255.255.0
    call drvMngGetConsoleLogLevel failed , g_conLogLevel = 3
    Device ID is out of the valid range.
    Cmd execute failed!

# Apparently the information in the manual is wrong or out-of-date, not sure what the device ID is then... ::

    [root@ml02 snassyr]# for i in {0..19}; do LD_PRELOAD=./libdrvdsmi_host.so:./libascend_hal.so ./hccn_tool -i $i -ip -s address 10.45.0.$(($i+1)) netmask 255.255.255.0; done
    call drvMngGetConsoleLogLevel failed , g_conLogLevel = 3
    Command execute failed!
    call drvMngGetConsoleLogLevel failed , g_conLogLevel = 3
    Command execute failed!
    call drvMngGetConsoleLogLevel failed , g_conLogLevel = 3
    Command execute failed!
    call drvMngGetConsoleLogLevel failed , g_conLogLevel = 3
    Command execute failed!
    call drvMngGetConsoleLogLevel failed , g_conLogLevel = 3
    Command execute failed!
    call drvMngGetConsoleLogLevel failed , g_conLogLevel = 3
    Command execute failed!
    call drvMngGetConsoleLogLevel failed , g_conLogLevel = 3
    Command execute failed!
    call drvMngGetConsoleLogLevel failed , g_conLogLevel = 3
    Command execute failed!
    call drvMngGetConsoleLogLevel failed , g_conLogLevel = 3
    Device ID is out of the valid range.
    Cmd execute failed!
    call drvMngGetConsoleLogLevel failed , g_conLogLevel = 3
    Device ID is out of the valid range.
    Cmd execute failed!
    call drvMngGetConsoleLogLevel failed , g_conLogLevel = 3
    Device ID is out of the valid range.
    Cmd execute failed!
    call drvMngGetConsoleLogLevel failed , g_conLogLevel = 3
    Device ID is out of the valid range.
    Cmd execute failed!
    call drvMngGetConsoleLogLevel failed , g_conLogLevel = 3
    Device ID is out of the valid range.
    Cmd execute failed!
    call drvMngGetConsoleLogLevel failed , g_conLogLevel = 3
    Device ID is out of the valid range.
    Cmd execute failed!
    call drvMngGetConsoleLogLevel failed , g_conLogLevel = 3
    Device ID is out of the valid range.
    Cmd execute failed!
    call drvMngGetConsoleLogLevel failed , g_conLogLevel = 3
    Device ID is out of the valid range.
    Cmd execute failed!
    call drvMngGetConsoleLogLevel failed , g_conLogLevel = 3
    Device ID is out of the valid range.
    Cmd execute failed!
    call drvMngGetConsoleLogLevel failed , g_conLogLevel = 3
    Device ID is out of the valid range.
    Cmd execute failed!
    call drvMngGetConsoleLogLevel failed , g_conLogLevel = 3
    Device ID is out of the valid range.
    Cmd execute failed!
    call drvMngGetConsoleLogLevel failed , g_conLogLevel = 3
    Device ID is out of the valid range.
    Cmd execute failed!

# So probably we have devices 0-7... wait are the devices not detected at runtime and are HARDCODED in the hccn_tool application?? Anyhow this seems to have failed.. I think this hccn_tool is device-specific so the 910A hccn_tool doesn't work with 310 npus... Abandoning ml02 here

Failing example
~~~~~~~~~~~~~~~

Code of the failing example:

.. code-block:: python

    [snassyr@login ZhongMNIST]$ cat npu_mnist2.py 
    #!/usr/bin/env python
    # coding: utf-8


    from __future__ import print_function
    from __future__ import absolute_import
    from __future__ import division
    import gzip
    import os
    import shutil
    import tempfile
    import numpy as np
    from six.moves import urllib
    import tensorflow as tf
    from npu_bridge.estimator.npu import util
    from npu_bridge.estimator.npu.npu_config import NPURunConfig
    from npu_bridge.estimator.npu.npu_estimator import NPUEstimator
    from tensorflow.contrib.layers import flatten
    from tensorflow.examples.tutorials.mnist import input_data


    data_dir = 'MNIST_data/'
    model_dir = 'saved_model/'
    learning_rate = 0.001
    batch_size = 256
    num_epochs = 1


    def read32(bytestream):
      """Read 4 bytes from bytestream as an unsigned 32-bit integer."""
      dt = np.dtype(np.uint32).newbyteorder('>')
      return np.frombuffer(bytestream.read(4), dtype=dt)[0]


    def check_image_file_header(filename):
      """Validate that filename corresponds to images for the MNIST dataset."""
      with tf.gfile.Open(filename, 'rb') as f:
        magic = read32(f)
        read32(f)  # num_images, unused
        rows = read32(f)
        cols = read32(f)
        if magic != 2051:
          raise ValueError('Invalid magic number %d in MNIST file %s' % (magic,
                                                                         f.name))
        if rows != 28 or cols != 28:
          raise ValueError(
              'Invalid MNIST file %s: Expected 28x28 images, found %dx%d' %
              (f.name, rows, cols))


    def check_labels_file_header(filename):
      """Validate that filename corresponds to labels for the MNIST dataset."""
      with tf.gfile.Open(filename, 'rb') as f:
        magic = read32(f)
        read32(f)  # num_items, unused
        if magic != 2049:
          raise ValueError('Invalid magic number %d in MNIST file %s' % (magic,
                                                                         f.name))


    def download(directory, filename):
      """Download (and unzip) a file from the MNIST dataset if not already done."""
      filepath = os.path.join(directory, filename)
      if tf.gfile.Exists(filepath):
        return filepath
      if not tf.gfile.Exists(directory):
        tf.gfile.MakeDirs(directory)
      # CVDF mirror of http://yann.lecun.com/exdb/mnist/
      url = 'https://storage.googleapis.com/cvdf-datasets/mnist/' + filename + '.gz'
      _, zipped_filepath = tempfile.mkstemp(suffix='.gz')
      print('Downloading %s to %s' % (url, zipped_filepath))
      urllib.request.urlretrieve(url, zipped_filepath)
      with gzip.open(zipped_filepath, 'rb') as f_in,       tf.gfile.Open(filepath, 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)
      os.remove(zipped_filepath)
      return filepath


    def dataset(directory, images_file, labels_file):
      """Download and parse MNIST dataset."""

      images_file = download(directory, images_file)
      labels_file = download(directory, labels_file)

      check_image_file_header(images_file)
      check_labels_file_header(labels_file)

      def decode_image(image):
        # Normalize from [0, 255] to [0.0, 1.0]
        image = tf.decode_raw(image, tf.uint8)
        image = tf.cast(image, tf.float32)
        image = tf.reshape(image, [784])
        return image / 255.0

      def decode_label(label):
        label = tf.decode_raw(label, tf.uint8)  # tf.string -> [tf.uint8]
        label = tf.reshape(label, [])  # label is a scalar
        return tf.to_int32(label)

      images = tf.data.FixedLengthRecordDataset(
          images_file, 28 * 28, header_bytes=16).map(decode_image)
      labels = tf.data.FixedLengthRecordDataset(
          labels_file, 1, header_bytes=8).map(decode_label)
      return tf.data.Dataset.zip((images, labels))


    def data_train(directory):
      """tf.data.Dataset object for MNIST training data."""
      return dataset(directory, 'train-images-idx3-ubyte',
                     'train-labels-idx1-ubyte')

    def data_test(directory):
      """tf.data.Dataset object for MNIST test data."""
      return dataset(directory, 't10k-images-idx3-ubyte', 't10k-labels-idx1-ubyte')


    def train_data():
        data = data_train(data_dir)
        data = data.cache().shuffle(buffer_size=50000).batch(128, drop_remainder=True) 
        return data


    def eval_data():
        data = data_test(data_dir)
        data = data.cache().shuffle(buffer_size=50000).batch(128, drop_remainder=True) 
        #data = data.batch(batch_size)
        return data


    def lenet():
        layers = tf.keras.layers

        model = tf.keras.Sequential([
            layers.Reshape(
                target_shape=[28, 28, 1],
                input_shape=(28 * 28,)),

            layers.Conv2D(
                filters=20,
                kernel_size=[5,5],
                padding='same',
                activation=tf.nn.relu),

            layers.MaxPooling2D(
                pool_size=[2,2]),

            layers.Conv2D(
                filters=50,
                kernel_size=[5,5],
                padding='same',
                activation=tf.nn.relu),

            layers.MaxPool2D(
                pool_size=[2,2]),

            layers.Flatten(),

            layers.Dense(
                units=500,
                activation=tf.nn.relu),

            layers.Dense(
                units=10),
        ])

        return model


    def model_function(features, labels, mode):
        
        model = lenet()
        
        if mode == tf.estimator.ModeKeys.TRAIN:
            # pass the input through the model
            logits = model(features)

            # get the cross-entropy loss and name it
            loss = tf.losses.sparse_softmax_cross_entropy(
                labels=labels,
                logits=logits)
            tf.identity(loss, 'train_loss')

            # record the accuracy and name it
            accuracy = tf.metrics.accuracy(
                labels=labels,
                predictions=tf.argmax(logits, axis=1))
            tf.identity(accuracy[1], name='train_accuracy')

            # use Adam to optimize
            optimizer = tf.train.AdamOptimizer(learning_rate)
            tf.identity(learning_rate, name='learning_rate')
            
            # create an estimator spec to optimize the loss
            estimator_spec = tf.estimator.EstimatorSpec(
                mode=tf.estimator.ModeKeys.TRAIN,
                loss=loss,
                train_op=optimizer.minimize(loss, tf.train.get_or_create_global_step()))

        elif mode == tf.estimator.ModeKeys.EVAL:
            # pass the input through the model
            logits = model(features, training=False)

            # get the cross-entropy loss
            loss = tf.losses.sparse_softmax_cross_entropy(
                labels=labels,
                logits=logits)

            # use the accuracy as a metric
            accuracy = tf.metrics.accuracy(
                labels=labels,
                predictions=tf.argmax(logits, axis=1))

            # create an estimator spec with the loss and accuracy
            estimator_spec = tf.estimator.EstimatorSpec(
                mode=tf.estimator.ModeKeys.EVAL,
                loss=loss,
                eval_metric_ops={
                    'accuracy': accuracy
                })

        return estimator_spec


    def main():
        hooks = [
            tf.train.LoggingTensorHook(
                ['train_accuracy', 'train_loss'],
                every_n_iter=1000)
        ]
        npu_config = NPURunConfig(
            model_dir = model_dir,
            save_checkpoints_steps = 10000,
            session_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        )
        
        mnist_classifier = NPUEstimator(
            model_fn=model_function,
            config = npu_config,
            model_dir=model_dir)
        for _ in range(num_epochs):
            mnist_classifier.train(
                input_fn=train_data,
                hooks=hooks,
            )
            mnist_classifier.evaluate(
                input_fn=eval_data)
            

    main()


    import tensorflow as tf
    from tensorflow.python.framework import graph_util
    from tensorflow.python import pywrap_tensorflow


    def freeze_graph(cpkt_path, pb_path):
        # checkpoint = tf.train.get_checkpoint_state("test1/model2/") #verify ckpt
        # cpkt_path2 = checkpoint.model_checkpoint_path #
        # cpkt_path3 = checkpoint.all_model_checkpoint_paths
        # print("model_pa:",cpkt_path3)

        # 
        # output_node_names = ["num_detections", "raw_detection_scores", "raw_detection_scores"]
        # output_node_names = "num_detections,raw_detection_boxes,raw_detection_scores"
        output_node_names = "dense/bias"
        saver = tf.train.import_meta_graph(cpkt_path + '.meta', clear_devices=True)
        graph = tf.get_default_graph()
        input_graph_def = graph.as_graph_def()

        # feature_data_list = input_graph_def.get_operation_by_name('resnet_v2_50/conv1').outputs[0]
        # input_image=tf.placeholder(None,28,28,1)

        with tf.Session() as sess:
            saver.restore(sess, cpkt_path)  # 

            pb_path_def = graph_util.convert_variables_to_constants(  # 
                sess=sess,
                input_graph_def=input_graph_def,  # equal:sess.graph_def
                output_node_names=output_node_names.split(","))  # spilt
            # print(pb_path_def)


            with tf.gfile.GFile(pb_path, 'wb') as fgraph:
                fgraph.write(pb_path_def.SerializeToString())
            # with tf.io.gfile.GFile(pb_path, "wb") as f:  # save 
            #     f.write(pb_path_def.SerializeToString())  # output sequence
            print("%d ops in the final graph." % len(pb_path_def.node))  # 
        

    """
    cpkt_path = 'saved_model/model.ckpt-2340'
        # pb_path_def
    pb_path = "saved_model/model.ckpt-2340.pb"

    reader = pywrap_tensorflow.NewCheckpointReader(cpkt_path)
    var_to_shape_map = reader.get_variable_to_shape_map()
    for key in var_to_shape_map:
        print("tensor_name: ", key)

    # model transformation
    freeze_graph(cpkt_path, pb_path)

    """


Running the failing example::

    $ python3 npu_mnist2.py
    WARNING:tensorflow:From npu_mnist2.py:278: The name tf.train.LoggingTensorHook is deprecated. Please use tf.estimator.LoggingTensorHook instead.

    WARNING:tensorflow:From npu_mnist2.py:285: The name tf.ConfigProto is deprecated. Please use tf.compat.v1.ConfigProto instead.

    WARNING:tensorflow:Warning:job config file does not exist
    WARNING:tensorflow:From /software/kp920-RL8/Stages/2021a/software/TensorFlow-CANN-Core/1.15.0-goolf-2021a.9-Python-3.7.5/lib/python3.7/site-packages/tensorflow_core/python/training/training_util.py:236: Variable.initialized_value (from tensorflow.python.ops.variables) is deprecated and will be removed in a future version.
    Instructions for updating:
    Use Variable.read_value. Variables in 2.X are initialized automatically both in eager and graph (inside tf.defun) contexts.
    WARNING:tensorflow:From npu_mnist2.py:72: The name tf.gfile.Exists is deprecated. Please use tf.io.gfile.exists instead.

    WARNING:tensorflow:From npu_mnist2.py:45: The name tf.gfile.Open is deprecated. Please use tf.io.gfile.GFile instead.

    WARNING:tensorflow:From npu_mnist2.py:106: to_int32 (from tensorflow.python.ops.math_ops) is deprecated and will be removed in a future version.
    Instructions for updating:
    Use `tf.cast` instead.
    WARNING:tensorflow:From /software/kp920-RL8/Stages/2021a/software/TensorFlow-CANN-Core/1.15.0-goolf-2021a.9-Python-3.7.5/lib/python3.7/site-packages/tensorflow_core/python/ops/resource_variable_ops.py:1630: calling BaseResourceVariable.__init__ (from tensorflow.python.ops.resource_variable_ops) with constraint is deprecated and will be removed in a future version.
    Instructions for updating:
    If using Keras pass *_constraint arguments to layers.
    WARNING:tensorflow:From npu_mnist2.py:221: The name tf.losses.sparse_softmax_cross_entropy is deprecated. Please use tf.compat.v1.losses.sparse_softmax_cross_entropy instead.

    WARNING:tensorflow:From /software/kp920-RL8/Stages/2021a/software/TensorFlow-CANN-Core/1.15.0-goolf-2021a.9-Python-3.7.5/lib/python3.7/site-packages/tensorflow_core/python/ops/losses/losses_impl.py:121: where (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
    Instructions for updating:
    Use tf.where in 2.0, which has the same broadcast rule as np.where
    WARNING:tensorflow:From npu_mnist2.py:227: The name tf.metrics.accuracy is deprecated. Please use tf.compat.v1.metrics.accuracy instead.

    WARNING:tensorflow:From npu_mnist2.py:233: The name tf.train.AdamOptimizer is deprecated. Please use tf.compat.v1.train.AdamOptimizer instead.

    WARNING:tensorflow:From npu_mnist2.py:240: The name tf.train.get_or_create_global_step is deprecated. Please use tf.compat.v1.train.get_or_create_global_step instead.

    WARNING:tensorflow:From /software/kp920-RL8/Stages/2021a/software/CANN-tfplugin/1.7.0-goolf-2021a.9-Python-3.7.5/lib/python/site-packages/npu_bridge/estimator/npu/npu_hook.py:158: The name tf.trainable_variables is deprecated. Please use tf.compat.v1.trainable_variables instead.

    WARNING:tensorflow:From /software/kp920-RL8/Stages/2021a/software/CANN-tfplugin/1.7.0-goolf-2021a.9-Python-3.7.5/lib/python/site-packages/npu_bridge/estimator/npu/npu_hook.py:164: The name tf.assign is deprecated. Please use tf.compat.v1.assign instead.

    2022-04-02 11:09:29.819786: I /dev/shm/snassyr/Kunpeng920/CANNtfplugin/1.7.0/goolf-2021a.9-Python-3.7.5/tensorflow/tf_adapter/kernels/geop_npu.cc:712] The model has been compiled on the Ascend AI processor, current graph id is:1
    Traceback (most recent call last):
      File "/software/kp920-RL8/Stages/2021a/software/TensorFlow-CANN-Core/1.15.0-goolf-2021a.9-Python-3.7.5/lib/python3.7/site-packages/tensorflow_core/python/client/session.py", line 1365, in _do_call
        return fn(*args)
      File "/software/kp920-RL8/Stages/2021a/software/TensorFlow-CANN-Core/1.15.0-goolf-2021a.9-Python-3.7.5/lib/python3.7/site-packages/tensorflow_core/python/client/session.py", line 1350, in _run_fn
        target_list, run_metadata)
      File "/software/kp920-RL8/Stages/2021a/software/TensorFlow-CANN-Core/1.15.0-goolf-2021a.9-Python-3.7.5/lib/python3.7/site-packages/tensorflow_core/python/client/session.py", line 1443, in _call_tf_sessionrun
        run_metadata)
    tensorflow.python.framework.errors_impl.InternalError: GeOp1_0GEOP::::DoRunAsync Failed
    Error Message is :
    EE9999: Inner Error!
            ctx is NULL![FUNC:HostMalloc][FILE:api_impl.cc][LINE:761]
            Host memory malloc failed, size=16.[FUNC:HostMalloc][FILE:logger.cc][LINE:354]
            rtMallocHost execute failed, reason=[null context pointer][FUNC:ReportFuncErrorReason][FILE:error_message_manage.cc][LINE:26]
            [GenTask][CreatHandle] alloc handle entity failed[FUNC:CreateHandle][FILE:task_builder.cc][LINE:56]
            [GenTask][CreateHandle][Node global_step/Assign type Assign] CreateHandle failed. ret:0xFFFFFFFF[FUNC:GenerateTask][FILE:task_builder.cc][LINE:119]
            Call OpsKernelBuilderManager GenerateTask fail for op:global_step/Assign(Assign)[FUNC:GenerateTask][FILE:task_generator.cc][LINE:421]

             [[{{node GeOp1_0}}]]

    During handling of the above exception, another exception occurred:

    Traceback (most recent call last):
      File "npu_mnist2.py", line 311, in <module>
        main()
      File "npu_mnist2.py", line 295, in main
        hooks=hooks,
      File "/software/kp920-RL8/Stages/2021a/software/TensorFlow-CANN-Core/1.15.0-goolf-2021a.9-Python-3.7.5/lib/python3.7/site-packages/tensorflow_estimator/python/estimator/estimator.py", line 370, in train
        loss = self._train_model(input_fn, hooks, saving_listeners)
      File "/software/kp920-RL8/Stages/2021a/software/TensorFlow-CANN-Core/1.15.0-goolf-2021a.9-Python-3.7.5/lib/python3.7/site-packages/tensorflow_estimator/python/estimator/estimator.py", line 1161, in _train_model
        return self._train_model_default(input_fn, hooks, saving_listeners)
      File "/software/kp920-RL8/Stages/2021a/software/TensorFlow-CANN-Core/1.15.0-goolf-2021a.9-Python-3.7.5/lib/python3.7/site-packages/tensorflow_estimator/python/estimator/estimator.py", line 1195, in _train_model_default
        saving_listeners)
      File "/software/kp920-RL8/Stages/2021a/software/TensorFlow-CANN-Core/1.15.0-goolf-2021a.9-Python-3.7.5/lib/python3.7/site-packages/tensorflow_estimator/python/estimator/estimator.py", line 1490, in _train_with_estimator_spec
        log_step_count_steps=log_step_count_steps) as mon_sess:
      File "/software/kp920-RL8/Stages/2021a/software/TensorFlow-CANN-Core/1.15.0-goolf-2021a.9-Python-3.7.5/lib/python3.7/site-packages/tensorflow_core/python/training/monitored_session.py", line 584, in MonitoredTrainingSession
        stop_grace_period_secs=stop_grace_period_secs)
      File "/software/kp920-RL8/Stages/2021a/software/TensorFlow-CANN-Core/1.15.0-goolf-2021a.9-Python-3.7.5/lib/python3.7/site-packages/tensorflow_core/python/training/monitored_session.py", line 1014, in __init__
        stop_grace_period_secs=stop_grace_period_secs)
      File "/software/kp920-RL8/Stages/2021a/software/TensorFlow-CANN-Core/1.15.0-goolf-2021a.9-Python-3.7.5/lib/python3.7/site-packages/tensorflow_core/python/training/monitored_session.py", line 725, in __init__
        self._sess = _RecoverableSession(self._coordinated_creator)
      File "/software/kp920-RL8/Stages/2021a/software/TensorFlow-CANN-Core/1.15.0-goolf-2021a.9-Python-3.7.5/lib/python3.7/site-packages/tensorflow_core/python/training/monitored_session.py", line 1207, in __init__
        _WrappedSession.__init__(self, self._create_session())
      File "/software/kp920-RL8/Stages/2021a/software/TensorFlow-CANN-Core/1.15.0-goolf-2021a.9-Python-3.7.5/lib/python3.7/site-packages/tensorflow_core/python/training/monitored_session.py", line 1212, in _create_session
        return self._sess_creator.create_session()
      File "/software/kp920-RL8/Stages/2021a/software/TensorFlow-CANN-Core/1.15.0-goolf-2021a.9-Python-3.7.5/lib/python3.7/site-packages/tensorflow_core/python/training/monitored_session.py", line 878, in create_session
        self.tf_sess = self._session_creator.create_session()
      File "/software/kp920-RL8/Stages/2021a/software/TensorFlow-CANN-Core/1.15.0-goolf-2021a.9-Python-3.7.5/lib/python3.7/site-packages/tensorflow_core/python/training/monitored_session.py", line 647, in create_session
        init_fn=self._scaffold.init_fn)
      File "/software/kp920-RL8/Stages/2021a/software/TensorFlow-CANN-Core/1.15.0-goolf-2021a.9-Python-3.7.5/lib/python3.7/site-packages/tensorflow_core/python/training/session_manager.py", line 296, in prepare_session
        sess.run(init_op, feed_dict=init_feed_dict)
      File "/software/kp920-RL8/Stages/2021a/software/TensorFlow-CANN-Core/1.15.0-goolf-2021a.9-Python-3.7.5/lib/python3.7/site-packages/tensorflow_core/python/client/session.py", line 956, in run
        run_metadata_ptr)
      File "/software/kp920-RL8/Stages/2021a/software/TensorFlow-CANN-Core/1.15.0-goolf-2021a.9-Python-3.7.5/lib/python3.7/site-packages/tensorflow_core/python/client/session.py", line 1180, in _run
        feed_dict_tensor, options, run_metadata)
      File "/software/kp920-RL8/Stages/2021a/software/TensorFlow-CANN-Core/1.15.0-goolf-2021a.9-Python-3.7.5/lib/python3.7/site-packages/tensorflow_core/python/client/session.py", line 1359, in _do_run
        run_metadata)
      File "/software/kp920-RL8/Stages/2021a/software/TensorFlow-CANN-Core/1.15.0-goolf-2021a.9-Python-3.7.5/lib/python3.7/site-packages/tensorflow_core/python/client/session.py", line 1384, in _do_call
        raise type(e)(node_def, op, message)
    tensorflow.python.framework.errors_impl.InternalError: GeOp1_0GEOP::::DoRunAsync Failed
    Error Message is :
    EE9999: Inner Error!
            ctx is NULL![FUNC:HostMalloc][FILE:api_impl.cc][LINE:761]
            Host memory malloc failed, size=16.[FUNC:HostMalloc][FILE:logger.cc][LINE:354]
            rtMallocHost execute failed, reason=[null context pointer][FUNC:ReportFuncErrorReason][FILE:error_message_manage.cc][LINE:26]
            [GenTask][CreatHandle] alloc handle entity failed[FUNC:CreateHandle][FILE:task_builder.cc][LINE:56]
            [GenTask][CreateHandle][Node global_step/Assign type Assign] CreateHandle failed. ret:0xFFFFFFFF[FUNC:GenerateTask][FILE:task_builder.cc][LINE:119]
            Call OpsKernelBuilderManager GenerateTask fail for op:global_step/Assign(Assign)[FUNC:GenerateTask][FILE:task_generator.cc][LINE:421]

             [[{{node GeOp1_0}}]]

Environment when running the failing example::

    $ env
    ARCHITECTURE=Kunpeng920
    BASH_ENV=/opt/ohpc/admin/lmod/lmod/init/bash
    BASH_FUNC_ml%%=() {  eval $($LMOD_DIR/ml_cmd "$@")
    }
    BASH_FUNC_module%%=() {  eval $($LMOD_CMD bash "$@") && eval $(${LMOD_SETTARG_CMD:-:} -s sh)
    }
    BASH_FUNC_which%%=() {  ( alias;
     eval ${which_declare} ) | /usr/bin/which --tty-only --read-alias --read-functions --show-tilde --show-dot "$@"
    }
    CMAKE_PREFIX_PATH=/software/kp920-RL8/Stages/2021a/software/PAPI/6.0.0.1-70887df7-GCCcore-9.3.0:/software/kp920-RL8/Stages/2021a/software/libpfm/4.11.1-f6500e77-GCCcore-9.3.0:/software/kp920-RL8/Stages/2021a/software/ncurses/6.2-GCCcore-9.3.0:/software/kp920-RL8/Stages/2021a/software/CANN-tfplugin/1.7.0-goolf-2021a.9-Python-3.7.5:/software/kp920-RL8/Stages/2021a/software/CANN-Toolkit/5.0.2.alpha005-goolf-2021a.9-Python-3.7.5:/software/kp920-RL8/Stages/2021a/software/CMake/3.20.0-GCCcore-9.3.0:/software/kp920-RL8/Stages/2021a/software/TensorFlow-CANN-Core/1.15.0-goolf-2021a.9-Python-3.7.5:/software/kp920-RL8/Stages/2021a/software/nsync-CANN/1.22.0-GCCcore-9.3.0:/software/kp920-RL8/Stages/2021a/software/h5py/3.5.0-goolf-2021a.9-Python-3.7.5:/software/kp920-RL8/Stages/2021a/software/HDF5/1.12.0-goolf-2021a.9:/software/kp920-RL8/Stages/2021a/software/Szip/2.1.1-GCCcore-9.3.0:/software/kp920-RL8/Stages/2021a/software/protobuf-CANN/3.15.6-GCCcore-9.3.0:/software/kp920-RL8/Stages/2021a/software/SciPy-Stack/2021a-goolf-2021a.9-Python-3.7.5:/software/kp920-RL8/Stages/2021a/software/pybind11/2.6.2-GCCcore-9.3.0-Python-3.7.5:/software/kp920-RL8/Stages/2021a/software/Python/3.7.5-GCCcore-9.3.0:/software/kp920-RL8/Stages/2021a/software/gRPC/1.37.1-GCCcore-9.3.0:/software/kp920-RL8/Stages/2021a/software/re2/2021-04-01-GCCcore-9.3.0:/software/kp920-RL8/Stages/2021a/software/c-ares/1.17.1-GCCcore-9.3.0:/software/kp920-RL8/Stages/2021a/software/abseil-cpp/20210324.1-GCCcore-9.3.0:/software/kp920-RL8/Stages/2021a/software/libjpeg-turbo/2.0.6-GCCcore-9.3.0:/software/kp920-RL8/Stages/2021a/software/NASM/2.15.05-GCCcore-9.3.0:/software/kp920-RL8/Stages/2021a/software/libspatialindex/1.9.3-GCCcore-9.3.0:/software/kp920-RL8/Stages/2021a/software/gflags/2.2.2-GCCcore-9.3.0:/software/kp920-RL8/Stages/2021a/software/protobuf/3.15.6-GCCcore-9.3.0:/software/kp920-RL8/Stages/2021a/software/Rust/1.52.1-GCCcore-9.3.0:/software/kp920-RL8/Stages/2021a/software/PostgreSQL/13.2-GCCcore-9.3.0:/software/kp920-RL8/Stages/2021a/software/Java/11.0.10:/software/kp920-RL8/Stages/2021a/software/libyaml/0.2.5-GCCcore-9.3.0:/software/kp920-RL8/Stages/2021a/software/libffi/3.3-GCCcore-9.3.0:/software/kp920-RL8/Stages/2021a/software/libxslt/1.1.34-GCCcore-9.3.0:/software/kp920-RL8/Stages/2021a/software/GMP/6.2.1-GCCcore-9.3.0:/software/kp920-RL8/Stages/2021a/software/SQLite/3.35.3-GCCcore-9.3.0:/software/kp920-RL8/Stages/2021a/software/Tcl/8.6.11-GCCcore-9.3.0:/software/kp920-RL8/Stages/2021a/software/libreadline/8.1-GCCcore-9.3.0:/software/kp920-RL8/Stages/2021a/software/ScaLAPACK/2.1.0-gompi-2021a.9-OpenBLAS-0.3.19:/software/kp920-RL8/Stages/2021a/software/FFTW/3.3.9-gompi-2021a.9:/software/kp920-RL8/Stages/2021a/software/OpenBLAS/0.3.19-GCC-9.3.0:/software/kp920-RL8/Stages/2021a/software/OpenMPI/4.1.2-GCC-9.3.0:/software/kp920-RL8/Stages/2021a/software/UCX/1.11.2-GCCcore-9.3.0:/software/kp920-RL8/Stages/2021a/software/hwloc/2.4.1-GCCcore-9.3.0:/software/kp920-RL8/Stages/2021a/software/X11/20210331-GCCcore-9.3.0:/software/kp920-RL8/Stages/2021a/software/fontconfig/2.13.93-GCCcore-9.3.0:/software/kp920-RL8/Stages/2021a/software/util-linux/2.35.1-GCCcore-9.3.0:/software/kp920-RL8/Stages/2021a/software/gperf/3.1-GCCcore-9.3.0:/software/kp920-RL8/Stages/2021a/software/freetype/2.10.4-GCCcore-9.3.0:/software/kp920-RL8/Stages/2021a/software/libpng/1.6.37-GCCcore-9.3.0:/software/kp920-RL8/Stages/2021a/software/expat/2.4.2-GCCcore-9.3.0:/software/kp920-RL8/Stages/2021a/software/bzip2/1.0.8-GCCcore-9.3.0:/software/kp920-RL8/Stages/2021a/software/libxml2/2.9.10-babe7503-GCCcore-9.3.0:/software/kp920-RL8/Stages/2021a/software/XZ/5.2.5-GCCcore-9.3.0:/software/kp920-RL8/Stages/2021a/software/numactl/2.0.14-GCCcore-9.3.0:/software/kp920-RL8/Stages/2021a/software/zlib/1.2.11-GCCcore-9.3.0:/software/kp920-RL8/Stages/2021a/software/binutils/2.36.1-GCCcore-9.3.0:/software/kp920-RL8/Stages/2021a/software/GCCcore/9.3.0:/software/kp920-RL8/Stages/2021a/software/zsh/5.8:/software/kp920-RL8/Stages/2021a/software/tmux/3.2a
    CPATH=/software/kp920-RL8/Stages/2021a/software/PAPI/6.0.0.1-70887df7-GCCcore-9.3.0/include:/software/kp920-RL8/Stages/2021a/software/libpfm/4.11.1-f6500e77-GCCcore-9.3.0/include:/software/kp920-RL8/Stages/2021a/software/ncurses/6.2-GCCcore-9.3.0/include:/software/kp920-RL8/Stages/2021a/software/CANN-Toolkit/5.0.2.alpha005-goolf-2021a.9-Python-3.7.5/ascend-toolkit/5.0.2.alpha005/acllib/include:/software/kp920-RL8/Stages/2021a/software/nsync-CANN/1.22.0-GCCcore-9.3.0/include:/software/kp920-RL8/Stages/2021a/software/HDF5/1.12.0-goolf-2021a.9/include:/software/kp920-RL8/Stages/2021a/software/Szip/2.1.1-GCCcore-9.3.0/include:/software/kp920-RL8/Stages/2021a/software/protobuf-CANN/3.15.6-GCCcore-9.3.0/include:/software/kp920-RL8/Stages/2021a/software/SciPy-Stack/2021a-goolf-2021a.9-Python-3.7.5/lib/python3.7/site-packages/numpy/core/include:/software/kp920-RL8/Stages/2021a/software/pybind11/2.6.2-GCCcore-9.3.0-Python-3.7.5/include:/software/kp920-RL8/Stages/2021a/software/Python/3.7.5-GCCcore-9.3.0/include:/software/kp920-RL8/Stages/2021a/software/gRPC/1.37.1-GCCcore-9.3.0/include:/software/kp920-RL8/Stages/2021a/software/re2/2021-04-01-GCCcore-9.3.0/include:/software/kp920-RL8/Stages/2021a/software/c-ares/1.17.1-GCCcore-9.3.0/include:/software/kp920-RL8/Stages/2021a/software/abseil-cpp/20210324.1-GCCcore-9.3.0/include:/software/kp920-RL8/Stages/2021a/software/libjpeg-turbo/2.0.6-GCCcore-9.3.0/include:/software/kp920-RL8/Stages/2021a/software/libspatialindex/1.9.3-GCCcore-9.3.0/include:/software/kp920-RL8/Stages/2021a/software/gflags/2.2.2-GCCcore-9.3.0/include:/software/kp920-RL8/Stages/2021a/software/protobuf/3.15.6-GCCcore-9.3.0/include:/software/kp920-RL8/Stages/2021a/software/PostgreSQL/13.2-GCCcore-9.3.0/include:/software/kp920-RL8/Stages/2021a/software/Java/11.0.10/include:/software/kp920-RL8/Stages/2021a/software/libyaml/0.2.5-GCCcore-9.3.0/include:/software/kp920-RL8/Stages/2021a/software/libffi/3.3-GCCcore-9.3.0/include:/software/kp920-RL8/Stages/2021a/software/libxslt/1.1.34-GCCcore-9.3.0/include:/software/kp920-RL8/Stages/2021a/software/GMP/6.2.1-GCCcore-9.3.0/include:/software/kp920-RL8/Stages/2021a/software/SQLite/3.35.3-GCCcore-9.3.0/include:/software/kp920-RL8/Stages/2021a/software/Tcl/8.6.11-GCCcore-9.3.0/include:/software/kp920-RL8/Stages/2021a/software/libreadline/8.1-GCCcore-9.3.0/include:/software/kp920-RL8/Stages/2021a/software/ScaLAPACK/2.1.0-gompi-2021a.9-OpenBLAS-0.3.19/include:/software/kp920-RL8/Stages/2021a/software/FFTW/3.3.9-gompi-2021a.9/include:/software/kp920-RL8/Stages/2021a/software/OpenBLAS/0.3.19-GCC-9.3.0/include:/software/kp920-RL8/Stages/2021a/software/OpenMPI/4.1.2-GCC-9.3.0/include:/software/kp920-RL8/Stages/2021a/software/UCX/1.11.2-GCCcore-9.3.0/include:/software/kp920-RL8/Stages/2021a/software/hwloc/2.4.1-GCCcore-9.3.0/include:/software/kp920-RL8/Stages/2021a/software/X11/20210331-GCCcore-9.3.0/include:/software/kp920-RL8/Stages/2021a/software/fontconfig/2.13.93-GCCcore-9.3.0/include:/software/kp920-RL8/Stages/2021a/software/util-linux/2.35.1-GCCcore-9.3.0/include:/software/kp920-RL8/Stages/2021a/software/freetype/2.10.4-GCCcore-9.3.0/include/freetype2:/software/kp920-RL8/Stages/2021a/software/libpng/1.6.37-GCCcore-9.3.0/include:/software/kp920-RL8/Stages/2021a/software/expat/2.4.2-GCCcore-9.3.0/include:/software/kp920-RL8/Stages/2021a/software/bzip2/1.0.8-GCCcore-9.3.0/include:/software/kp920-RL8/Stages/2021a/software/libxml2/2.9.10-babe7503-GCCcore-9.3.0/include/libxml2:/software/kp920-RL8/Stages/2021a/software/libxml2/2.9.10-babe7503-GCCcore-9.3.0/include:/software/kp920-RL8/Stages/2021a/software/XZ/5.2.5-GCCcore-9.3.0/include:/software/kp920-RL8/Stages/2021a/software/numactl/2.0.14-GCCcore-9.3.0/include:/software/kp920-RL8/Stages/2021a/software/zlib/1.2.11-GCCcore-9.3.0/include:/software/kp920-RL8/Stages/2021a/software/binutils/2.36.1-GCCcore-9.3.0/include:/software/libs/arm-optimized-routines/21.02/Kunpeng920/usr/include
    DBUS_SESSION_BUS_ADDRESS=unix:path=/run/user/1111800007/bus
    EASYCONFIGREPO=/home/snassyr/SourceCode/git/juawei-easyconfigs
    EBDEVELNCURSES=/software/kp920-RL8/Stages/2021a/software/ncurses/6.2-GCCcore-9.3.0/easybuild/Compiler-GCCcore-9.3.0-ncurses-.6.2-easybuild-devel
    EBDEVELTMUX=/software/kp920-RL8/Stages/2021a/software/tmux/3.2a/easybuild/Core-tmux-3.2a-easybuild-devel
    EBDEVELZSH=/software/kp920-RL8/Stages/2021a/software/zsh/5.8/easybuild/Core-zsh-5.8-easybuild-devel
    EBROOTNCURSES=/software/kp920-RL8/Stages/2021a/software/ncurses/6.2-GCCcore-9.3.0
    EBROOTTMUX=/software/kp920-RL8/Stages/2021a/software/tmux/3.2a
    EBROOTZSH=/software/kp920-RL8/Stages/2021a/software/zsh/5.8
    EBVERSIONNCURSES=6.2
    EBVERSIONTMUX=3.2a
    EBVERSIONZSH=5.8
    FPATH=/home/snassyr/.oh-my-zsh/plugins/git:/home/snassyr/.oh-my-zsh/functions:/home/snassyr/.oh-my-zsh/completions:/home/snassyr/.oh-my-zsh/cache/completions:/software/kp920-RL8/Stages/2021a/software/zsh/5.8/share/zsh/5.8/functions:/software/kp920-RL8/Stages/2021a/software/zsh/5.8/share/zsh/site-functions:/opt/ohpc/admin/lmod/lmod/init/ksh_funcs
    HISTCONTROL=ignoredups
    HISTSIZE=1000
    HOME=/home/snassyr
    HOSTNAME=ml01.guoehi.cluster
    LANG=en_US.UTF-8
    LD_LIBRARY_PATH=/software/kp920-RL8/Stages/2021a/software/PAPI/6.0.0.1-70887df7-GCCcore-9.3.0/lib:/software/kp920-RL8/Stages/2021a/software/libpfm/4.11.1-f6500e77-GCCcore-9.3.0/lib:/software/kp920-RL8/Stages/2021a/software/ncurses/6.2-GCCcore-9.3.0/lib:/software/kp920-RL8/Stages/2021a/software/CANN-Toolkit/5.0.2.alpha005-goolf-2021a.9-Python-3.7.5/ascend-toolkit/5.0.2.alpha005/atc/lib64:/software/kp920-RL8/Stages/2021a/software/CANN-Toolkit/5.0.2.alpha005-goolf-2021a.9-Python-3.7.5/ascend-toolkit/5.0.2.alpha005/fwkacllib/lib64:/software/kp920-RL8/Stages/2021a/software/nsync-CANN/1.22.0-GCCcore-9.3.0/lib:/software/kp920-RL8/Stages/2021a/software/HDF5/1.12.0-goolf-2021a.9/lib:/software/kp920-RL8/Stages/2021a/software/Szip/2.1.1-GCCcore-9.3.0/lib:/software/kp920-RL8/Stages/2021a/software/protobuf-CANN/3.15.6-GCCcore-9.3.0/lib:/software/kp920-RL8/Stages/2021a/software/SciPy-Stack/2021a-goolf-2021a.9-Python-3.7.5/lib/python3.7/site-packages/numpy/core/lib:/software/kp920-RL8/Stages/2021a/software/SciPy-Stack/2021a-goolf-2021a.9-Python-3.7.5/lib:/software/kp920-RL8/Stages/2021a/software/Python/3.7.5-GCCcore-9.3.0/lib:/software/kp920-RL8/Stages/2021a/software/gRPC/1.37.1-GCCcore-9.3.0/lib:/software/kp920-RL8/Stages/2021a/software/re2/2021-04-01-GCCcore-9.3.0/lib:/software/kp920-RL8/Stages/2021a/software/c-ares/1.17.1-GCCcore-9.3.0/lib:/software/kp920-RL8/Stages/2021a/software/abseil-cpp/20210324.1-GCCcore-9.3.0/lib:/software/kp920-RL8/Stages/2021a/software/libjpeg-turbo/2.0.6-GCCcore-9.3.0/lib:/software/kp920-RL8/Stages/2021a/software/libspatialindex/1.9.3-GCCcore-9.3.0/lib:/software/kp920-RL8/Stages/2021a/software/gflags/2.2.2-GCCcore-9.3.0/lib:/software/kp920-RL8/Stages/2021a/software/protobuf/3.15.6-GCCcore-9.3.0/lib:/software/kp920-RL8/Stages/2021a/software/Rust/1.52.1-GCCcore-9.3.0/lib:/software/kp920-RL8/Stages/2021a/software/PostgreSQL/13.2-GCCcore-9.3.0/lib:/software/kp920-RL8/Stages/2021a/software/Java/11.0.10/lib:/software/kp920-RL8/Stages/2021a/software/libyaml/0.2.5-GCCcore-9.3.0/lib:/software/kp920-RL8/Stages/2021a/software/libffi/3.3-GCCcore-9.3.0/lib64:/software/kp920-RL8/Stages/2021a/software/libffi/3.3-GCCcore-9.3.0/lib:/software/kp920-RL8/Stages/2021a/software/libxslt/1.1.34-GCCcore-9.3.0/lib:/software/kp920-RL8/Stages/2021a/software/GMP/6.2.1-GCCcore-9.3.0/lib:/software/kp920-RL8/Stages/2021a/software/SQLite/3.35.3-GCCcore-9.3.0/lib:/software/kp920-RL8/Stages/2021a/software/Tcl/8.6.11-GCCcore-9.3.0/lib:/software/kp920-RL8/Stages/2021a/software/libreadline/8.1-GCCcore-9.3.0/lib:/software/kp920-RL8/Stages/2021a/software/ScaLAPACK/2.1.0-gompi-2021a.9-OpenBLAS-0.3.19/lib:/software/kp920-RL8/Stages/2021a/software/FFTW/3.3.9-gompi-2021a.9/lib:/software/kp920-RL8/Stages/2021a/software/OpenBLAS/0.3.19-GCC-9.3.0/lib:/software/kp920-RL8/Stages/2021a/software/OpenMPI/4.1.2-GCC-9.3.0/lib:/software/kp920-RL8/Stages/2021a/software/UCX/1.11.2-GCCcore-9.3.0/lib:/software/kp920-RL8/Stages/2021a/software/hwloc/2.4.1-GCCcore-9.3.0/lib:/software/kp920-RL8/Stages/2021a/software/X11/20210331-GCCcore-9.3.0/lib:/software/kp920-RL8/Stages/2021a/software/fontconfig/2.13.93-GCCcore-9.3.0/lib:/software/kp920-RL8/Stages/2021a/software/util-linux/2.35.1-GCCcore-9.3.0/lib:/software/kp920-RL8/Stages/2021a/software/freetype/2.10.4-GCCcore-9.3.0/lib:/software/kp920-RL8/Stages/2021a/software/libpng/1.6.37-GCCcore-9.3.0/lib:/software/kp920-RL8/Stages/2021a/software/expat/2.4.2-GCCcore-9.3.0/lib:/software/kp920-RL8/Stages/2021a/software/bzip2/1.0.8-GCCcore-9.3.0/lib:/software/kp920-RL8/Stages/2021a/software/libxml2/2.9.10-babe7503-GCCcore-9.3.0/lib:/software/kp920-RL8/Stages/2021a/software/XZ/5.2.5-GCCcore-9.3.0/lib:/software/kp920-RL8/Stages/2021a/software/numactl/2.0.14-GCCcore-9.3.0/lib:/software/kp920-RL8/Stages/2021a/software/zlib/1.2.11-GCCcore-9.3.0/lib:/software/kp920-RL8/Stages/2021a/software/binutils/2.36.1-GCCcore-9.3.0/lib:/software/kp920-RL8/Stages/2021a/software/GCCcore/9.3.0/lib64:/software/kp920-RL8/Stages/2021a/software/GCCcore/9.3.0/lib:/software/libs/arm-optimized-routines/21.02/Kunpeng920/usr/lib:/software/kp920-RL8/Stages/2021a/software/zsh/5.8/lib
    LESSOPEN=||/usr/bin/lesspipe.sh %s
    LIBRARY_PATH=/software/kp920-RL8/Stages/2021a/software/PAPI/6.0.0.1-70887df7-GCCcore-9.3.0/lib:/software/kp920-RL8/Stages/2021a/software/libpfm/4.11.1-f6500e77-GCCcore-9.3.0/lib:/software/kp920-RL8/Stages/2021a/software/ncurses/6.2-GCCcore-9.3.0/lib:/software/kp920-RL8/Stages/2021a/software/CANN-tfplugin/1.7.0-goolf-2021a.9-Python-3.7.5/lib:/software/kp920-RL8/Stages/2021a/software/CANN-Toolkit/5.0.2.alpha005-goolf-2021a.9-Python-3.7.5/ascend-toolkit/5.0.2.alpha005/atc/lib64:/software/kp920-RL8/Stages/2021a/software/CANN-Toolkit/5.0.2.alpha005-goolf-2021a.9-Python-3.7.5/ascend-toolkit/5.0.2.alpha005/fwkacllib/lib64:/software/kp920-RL8/Stages/2021a/software/TensorFlow-CANN-Core/1.15.0-goolf-2021a.9-Python-3.7.5/lib:/software/kp920-RL8/Stages/2021a/software/nsync-CANN/1.22.0-GCCcore-9.3.0/lib:/software/kp920-RL8/Stages/2021a/software/HDF5/1.12.0-goolf-2021a.9/lib:/software/kp920-RL8/Stages/2021a/software/Szip/2.1.1-GCCcore-9.3.0/lib:/software/kp920-RL8/Stages/2021a/software/protobuf-CANN/3.15.6-GCCcore-9.3.0/lib:/software/kp920-RL8/Stages/2021a/software/SciPy-Stack/2021a-goolf-2021a.9-Python-3.7.5/lib/python3.7/site-packages/numpy/core/lib:/software/kp920-RL8/Stages/2021a/software/SciPy-Stack/2021a-goolf-2021a.9-Python-3.7.5/lib:/software/kp920-RL8/Stages/2021a/software/Python/3.7.5-GCCcore-9.3.0/lib:/software/kp920-RL8/Stages/2021a/software/gRPC/1.37.1-GCCcore-9.3.0/lib:/software/kp920-RL8/Stages/2021a/software/re2/2021-04-01-GCCcore-9.3.0/lib:/software/kp920-RL8/Stages/2021a/software/c-ares/1.17.1-GCCcore-9.3.0/lib:/software/kp920-RL8/Stages/2021a/software/abseil-cpp/20210324.1-GCCcore-9.3.0/lib:/software/kp920-RL8/Stages/2021a/software/libjpeg-turbo/2.0.6-GCCcore-9.3.0/lib:/software/kp920-RL8/Stages/2021a/software/libspatialindex/1.9.3-GCCcore-9.3.0/lib:/software/kp920-RL8/Stages/2021a/software/gflags/2.2.2-GCCcore-9.3.0/lib:/software/kp920-RL8/Stages/2021a/software/protobuf/3.15.6-GCCcore-9.3.0/lib:/software/kp920-RL8/Stages/2021a/software/Rust/1.52.1-GCCcore-9.3.0/lib:/software/kp920-RL8/Stages/2021a/software/PostgreSQL/13.2-GCCcore-9.3.0/lib:/software/kp920-RL8/Stages/2021a/software/Java/11.0.10/lib:/software/kp920-RL8/Stages/2021a/software/libyaml/0.2.5-GCCcore-9.3.0/lib:/software/kp920-RL8/Stages/2021a/software/libffi/3.3-GCCcore-9.3.0/lib64:/software/kp920-RL8/Stages/2021a/software/libffi/3.3-GCCcore-9.3.0/lib:/software/kp920-RL8/Stages/2021a/software/libxslt/1.1.34-GCCcore-9.3.0/lib:/software/kp920-RL8/Stages/2021a/software/GMP/6.2.1-GCCcore-9.3.0/lib:/software/kp920-RL8/Stages/2021a/software/SQLite/3.35.3-GCCcore-9.3.0/lib:/software/kp920-RL8/Stages/2021a/software/Tcl/8.6.11-GCCcore-9.3.0/lib:/software/kp920-RL8/Stages/2021a/software/libreadline/8.1-GCCcore-9.3.0/lib:/software/kp920-RL8/Stages/2021a/software/ScaLAPACK/2.1.0-gompi-2021a.9-OpenBLAS-0.3.19/lib:/software/kp920-RL8/Stages/2021a/software/FFTW/3.3.9-gompi-2021a.9/lib:/software/kp920-RL8/Stages/2021a/software/OpenBLAS/0.3.19-GCC-9.3.0/lib:/software/kp920-RL8/Stages/2021a/software/OpenMPI/4.1.2-GCC-9.3.0/lib:/software/kp920-RL8/Stages/2021a/software/UCX/1.11.2-GCCcore-9.3.0/lib:/software/kp920-RL8/Stages/2021a/software/hwloc/2.4.1-GCCcore-9.3.0/lib:/software/kp920-RL8/Stages/2021a/software/X11/20210331-GCCcore-9.3.0/lib:/software/kp920-RL8/Stages/2021a/software/fontconfig/2.13.93-GCCcore-9.3.0/lib:/software/kp920-RL8/Stages/2021a/software/util-linux/2.35.1-GCCcore-9.3.0/lib:/software/kp920-RL8/Stages/2021a/software/freetype/2.10.4-GCCcore-9.3.0/lib:/software/kp920-RL8/Stages/2021a/software/libpng/1.6.37-GCCcore-9.3.0/lib:/software/kp920-RL8/Stages/2021a/software/expat/2.4.2-GCCcore-9.3.0/lib:/software/kp920-RL8/Stages/2021a/software/bzip2/1.0.8-GCCcore-9.3.0/lib:/software/kp920-RL8/Stages/2021a/software/libxml2/2.9.10-babe7503-GCCcore-9.3.0/lib:/software/kp920-RL8/Stages/2021a/software/XZ/5.2.5-GCCcore-9.3.0/lib:/software/kp920-RL8/Stages/2021a/software/numactl/2.0.14-GCCcore-9.3.0/lib:/software/kp920-RL8/Stages/2021a/software/zlib/1.2.11-GCCcore-9.3.0/lib:/software/kp920-RL8/Stages/2021a/software/binutils/2.36.1-GCCcore-9.3.0/lib:/software/libs/arm-optimized-routines/21.02/Kunpeng920/usr/lib:/software/kp920-RL8/Stages/2021a/software/zsh/5.8/lib
    LMOD_AVAIL_STYLE=system:<en_grouped>
    LMOD_CMD=/opt/ohpc/admin/lmod/lmod/libexec/lmod
    LMOD_COLORIZE=no
    LMOD_DIR=/opt/ohpc/admin/lmod/lmod/libexec
    LMOD_FULL_SETTARG_SUPPORT=no
    LMOD_PACKAGE_PATH=/software
    LMOD_PKG=/opt/ohpc/admin/lmod/lmod
    LMOD_PREPEND_BLOCK=normal
    LMOD_RC=/software/kp920-RL8/configs/lmodrc.lua
    LMOD_ROOT=/opt/ohpc/admin/lmod
    LMOD_SETTARG_CMD=:
    LMOD_SETTARG_FULL_SUPPORT=no
    LMOD_SYSTEM_NAME=kp920-RL8
    LMOD_VERSION=8.5.22
    LOADEDMODULES=tmux/3.2a:zsh/5.8:arm-optimized-routines/21.02:Architecture/.Kunpeng920:GCCcore/.9.3.0:binutils/.2.36.1:GCC/9.3.0:zlib/.1.2.11:numactl/2.0.14:XZ/.5.2.5:libxml2/.2.9.10-babe7503:bzip2/.1.0.8:expat/.2.4.2:libpng/.1.6.37:freetype/.2.10.4:gperf/.3.1:util-linux/.2.35.1:fontconfig/.2.13.93:X11/20210331:hwloc/2.4.1:UCX/1.11.2:OpenMPI/4.1.2:OpenBLAS/0.3.19:FFTW/3.3.9:ScaLAPACK/2.1.0-OpenBLAS-0.3.19:libreadline/.8.1:Tcl/8.6.11:SQLite/.3.35.3:GMP/6.2.1:libxslt/.1.1.34:libffi/3.3:libyaml/0.2.5:Java/11.0.10:PostgreSQL/13.2:Rust/1.52.1:protobuf/.3.15.6:gflags/.2.2.2:libspatialindex/.1.9.3:NASM/.2.15.05:libjpeg-turbo/.2.0.6:abseil-cpp/20210324.1:c-ares/1.17.1:re2/2021-04-01:gRPC/1.37.1:Python/3.7.5:pybind11/.2.6.2-Python-3.7.5:SciPy-Stack/2021a-Python-3.7.5:protobuf-CANN/3.15.6:Szip/.2.1.1:HDF5/1.12.0:h5py/3.5.0-Python-3.7.5:nsync-CANN/1.22.0:TensorFlow-CANN-Core/1.15.0-Python-3.7.5:CMake/3.20.0:CANN-Toolkit/5.0.2.alpha005-Python-3.7.5:CANN-tfplugin/1.7.0-Python-3.7.5:TensorFlow-CANN/1.15.0-Python-3.7.5:ncurses/.6.2:libpfm/4.11.1-f6500e77:PAPI/6.0.0.1-70887df7
    LOGNAME=snassyr
    LS_COLORS=rs=0:di=01;34:ln=01;36:mh=00:pi=40;33:so=01;35:do=01;35:bd=40;33;01:cd=40;33;01:or=40;31;01:mi=01;05;37;41:su=37;41:sg=30;43:ca=30;41:tw=30;42:ow=34;42:st=37;44:ex=01;32:*.tar=01;31:*.tgz=01;31:*.arc=01;31:*.arj=01;31:*.taz=01;31:*.lha=01;31:*.lz4=01;31:*.lzh=01;31:*.lzma=01;31:*.tlz=01;31:*.txz=01;31:*.tzo=01;31:*.t7z=01;31:*.zip=01;31:*.z=01;31:*.dz=01;31:*.gz=01;31:*.lrz=01;31:*.lz=01;31:*.lzo=01;31:*.xz=01;31:*.zst=01;31:*.tzst=01;31:*.bz2=01;31:*.bz=01;31:*.tbz=01;31:*.tbz2=01;31:*.tz=01;31:*.deb=01;31:*.rpm=01;31:*.jar=01;31:*.war=01;31:*.ear=01;31:*.sar=01;31:*.rar=01;31:*.alz=01;31:*.ace=01;31:*.zoo=01;31:*.cpio=01;31:*.7z=01;31:*.rz=01;31:*.cab=01;31:*.wim=01;31:*.swm=01;31:*.dwm=01;31:*.esd=01;31:*.jpg=01;35:*.jpeg=01;35:*.mjpg=01;35:*.mjpeg=01;35:*.gif=01;35:*.bmp=01;35:*.pbm=01;35:*.pgm=01;35:*.ppm=01;35:*.tga=01;35:*.xbm=01;35:*.xpm=01;35:*.tif=01;35:*.tiff=01;35:*.png=01;35:*.svg=01;35:*.svgz=01;35:*.mng=01;35:*.pcx=01;35:*.mov=01;35:*.mpg=01;35:*.mpeg=01;35:*.m2v=01;35:*.mkv=01;35:*.webm=01;35:*.ogm=01;35:*.mp4=01;35:*.m4v=01;35:*.mp4v=01;35:*.vob=01;35:*.qt=01;35:*.nuv=01;35:*.wmv=01;35:*.asf=01;35:*.rm=01;35:*.rmvb=01;35:*.flc=01;35:*.avi=01;35:*.fli=01;35:*.flv=01;35:*.gl=01;35:*.dl=01;35:*.xcf=01;35:*.xwd=01;35:*.yuv=01;35:*.cgm=01;35:*.emf=01;35:*.ogv=01;35:*.ogx=01;35:*.aac=01;36:*.au=01;36:*.flac=01;36:*.m4a=01;36:*.mid=01;36:*.midi=01;36:*.mka=01;36:*.mp3=01;36:*.mpc=01;36:*.ogg=01;36:*.ra=01;36:*.wav=01;36:*.oga=01;36:*.opus=01;36:*.spx=01;36:*.xspf=01;36:
    MAIL=/var/spool/mail/snassyr
    MANPATH=/software/kp920-RL8/Stages/2021a/software/PAPI/6.0.0.1-70887df7-GCCcore-9.3.0/share/man:/software/kp920-RL8/Stages/2021a/software/libpfm/4.11.1-f6500e77-GCCcore-9.3.0/share/man:/software/kp920-RL8/Stages/2021a/software/ncurses/6.2-GCCcore-9.3.0/share/man:/software/kp920-RL8/Stages/2021a/software/SciPy-Stack/2021a-goolf-2021a.9-Python-3.7.5/share/man:/software/kp920-RL8/Stages/2021a/software/Python/3.7.5-GCCcore-9.3.0/share/man:/software/kp920-RL8/Stages/2021a/software/Python/3.7.5-GCCcore-9.3.0/man:/software/kp920-RL8/Stages/2021a/software/c-ares/1.17.1-GCCcore-9.3.0/share/man:/software/kp920-RL8/Stages/2021a/software/libjpeg-turbo/2.0.6-GCCcore-9.3.0/share/man:/software/kp920-RL8/Stages/2021a/software/NASM/2.15.05-GCCcore-9.3.0/share/man:/software/kp920-RL8/Stages/2021a/software/Rust/1.52.1-GCCcore-9.3.0/share/man:/software/kp920-RL8/Stages/2021a/software/Java/11.0.10/man:/software/kp920-RL8/Stages/2021a/software/libffi/3.3-GCCcore-9.3.0/share/man:/software/kp920-RL8/Stages/2021a/software/libxslt/1.1.34-GCCcore-9.3.0/share/man:/software/kp920-RL8/Stages/2021a/software/SQLite/3.35.3-GCCcore-9.3.0/share/man:/software/kp920-RL8/Stages/2021a/software/Tcl/8.6.11-GCCcore-9.3.0/share/man:/software/kp920-RL8/Stages/2021a/software/Tcl/8.6.11-GCCcore-9.3.0/man:/software/kp920-RL8/Stages/2021a/software/libreadline/8.1-GCCcore-9.3.0/share/man:/software/kp920-RL8/Stages/2021a/software/FFTW/3.3.9-gompi-2021a.9/share/man:/software/kp920-RL8/Stages/2021a/software/OpenMPI/4.1.2-GCC-9.3.0/share/man:/software/kp920-RL8/Stages/2021a/software/hwloc/2.4.1-GCCcore-9.3.0/share/man:/software/kp920-RL8/Stages/2021a/software/X11/20210331-GCCcore-9.3.0/share/man:/software/kp920-RL8/Stages/2021a/software/util-linux/2.35.1-GCCcore-9.3.0/share/man:/software/kp920-RL8/Stages/2021a/software/gperf/3.1-GCCcore-9.3.0/share/man:/software/kp920-RL8/Stages/2021a/software/freetype/2.10.4-GCCcore-9.3.0/share/man:/software/kp920-RL8/Stages/2021a/software/libpng/1.6.37-GCCcore-9.3.0/share/man:/software/kp920-RL8/Stages/2021a/software/bzip2/1.0.8-GCCcore-9.3.0/man:/software/kp920-RL8/Stages/2021a/software/libxml2/2.9.10-babe7503-GCCcore-9.3.0/share/man:/software/kp920-RL8/Stages/2021a/software/XZ/5.2.5-GCCcore-9.3.0/share/man:/software/kp920-RL8/Stages/2021a/software/numactl/2.0.14-GCCcore-9.3.0/share/man:/software/kp920-RL8/Stages/2021a/software/zlib/1.2.11-GCCcore-9.3.0/share/man:/software/kp920-RL8/Stages/2021a/software/binutils/2.36.1-GCCcore-9.3.0/share/man:/software/kp920-RL8/Stages/2021a/software/GCCcore/9.3.0/share/man:/software/kp920-RL8/Stages/2021a/software/zsh/5.8/share/man:/software/kp920-RL8/Stages/2021a/software/tmux/3.2a/share/man
    MATHLIB_PATH=/software/libs/arm-optimized-routines/21.02/Kunpeng920
    MODULEPATH=/software/kp920-RL8/Stages/2021a/modules/all/MPI/GCC/9.3.0/OpenMPI/4.1.2:/software/kp920-RL8/Stages/2021a/modules/all/Compiler/mpi/GCC/9.3.0:/software/kp920-RL8/Stages/2021a/modules/all/Compiler/GCC/9.3.0:/software/kp920-RL8/Stages/2021a/modules/all/Compiler/GCCcore/9.3.0:/software/kp920-RL8/Stages/2021a/UI/Compilers:/software/kp920-RL8/Stages/2021a/UI/Tools:/software/modules/system:/software/modules/custom
    MODULESHOME=/opt/ohpc/admin/lmod/lmod
    OTHERSTAGES=/software/kp920-RL8/OtherStages
    PATH=/software/kp920-RL8/Stages/2021a/software/PAPI/6.0.0.1-70887df7-GCCcore-9.3.0/bin:/software/kp920-RL8/Stages/2021a/software/ncurses/6.2-GCCcore-9.3.0/bin:/software/kp920-RL8/Stages/2021a/software/CANN-Toolkit/5.0.2.alpha005-goolf-2021a.9-Python-3.7.5:/software/kp920-RL8/Stages/2021a/software/CANN-Toolkit/5.0.2.alpha005-goolf-2021a.9-Python-3.7.5/ascend-toolkit/5.0.2.alpha005/atc/bin:/software/kp920-RL8/Stages/2021a/software/CANN-Toolkit/5.0.2.alpha005-goolf-2021a.9-Python-3.7.5/ascend-toolkit/5.0.2.alpha005/fwkacllib/bin:/software/kp920-RL8/Stages/2021a/software/CANN-Toolkit/5.0.2.alpha005-goolf-2021a.9-Python-3.7.5/ascend-toolkit/5.0.2.alpha005/fwkacllib/ccec_compiler/bin:/software/kp920-RL8/Stages/2021a/software/CMake/3.20.0-GCCcore-9.3.0/bin:/software/kp920-RL8/Stages/2021a/software/TensorFlow-CANN-Core/1.15.0-goolf-2021a.9-Python-3.7.5/bin:/software/kp920-RL8/Stages/2021a/software/HDF5/1.12.0-goolf-2021a.9/bin:/software/kp920-RL8/Stages/2021a/software/protobuf-CANN/3.15.6-GCCcore-9.3.0/bin:/software/kp920-RL8/Stages/2021a/software/SciPy-Stack/2021a-goolf-2021a.9-Python-3.7.5/bin:/software/kp920-RL8/Stages/2021a/software/pybind11/2.6.2-GCCcore-9.3.0-Python-3.7.5/bin:/software/kp920-RL8/Stages/2021a/software/Python/3.7.5-GCCcore-9.3.0/bin:/software/kp920-RL8/Stages/2021a/software/gRPC/1.37.1-GCCcore-9.3.0/bin:/software/kp920-RL8/Stages/2021a/software/c-ares/1.17.1-GCCcore-9.3.0/bin:/software/kp920-RL8/Stages/2021a/software/libjpeg-turbo/2.0.6-GCCcore-9.3.0/bin:/software/kp920-RL8/Stages/2021a/software/NASM/2.15.05-GCCcore-9.3.0/bin:/software/kp920-RL8/Stages/2021a/software/gflags/2.2.2-GCCcore-9.3.0/bin:/software/kp920-RL8/Stages/2021a/software/protobuf/3.15.6-GCCcore-9.3.0/bin:/software/kp920-RL8/Stages/2021a/software/Rust/1.52.1-GCCcore-9.3.0/bin:/software/kp920-RL8/Stages/2021a/software/PostgreSQL/13.2-GCCcore-9.3.0/bin:/software/kp920-RL8/Stages/2021a/software/Java/11.0.10:/software/kp920-RL8/Stages/2021a/software/Java/11.0.10/bin:/software/kp920-RL8/Stages/2021a/software/libxslt/1.1.34-GCCcore-9.3.0/bin:/software/kp920-RL8/Stages/2021a/software/SQLite/3.35.3-GCCcore-9.3.0/bin:/software/kp920-RL8/Stages/2021a/software/Tcl/8.6.11-GCCcore-9.3.0/bin:/software/kp920-RL8/Stages/2021a/software/FFTW/3.3.9-gompi-2021a.9/bin:/software/kp920-RL8/Stages/2021a/software/OpenMPI/4.1.2-GCC-9.3.0/bin:/software/kp920-RL8/Stages/2021a/software/UCX/1.11.2-GCCcore-9.3.0/bin:/software/kp920-RL8/Stages/2021a/software/hwloc/2.4.1-GCCcore-9.3.0/bin:/software/kp920-RL8/Stages/2021a/software/X11/20210331-GCCcore-9.3.0/bin:/software/kp920-RL8/Stages/2021a/software/fontconfig/2.13.93-GCCcore-9.3.0/bin:/software/kp920-RL8/Stages/2021a/software/util-linux/2.35.1-GCCcore-9.3.0/sbin:/software/kp920-RL8/Stages/2021a/software/util-linux/2.35.1-GCCcore-9.3.0/bin:/software/kp920-RL8/Stages/2021a/software/gperf/3.1-GCCcore-9.3.0/bin:/software/kp920-RL8/Stages/2021a/software/freetype/2.10.4-GCCcore-9.3.0/bin:/software/kp920-RL8/Stages/2021a/software/libpng/1.6.37-GCCcore-9.3.0/bin:/software/kp920-RL8/Stages/2021a/software/expat/2.4.2-GCCcore-9.3.0/bin:/software/kp920-RL8/Stages/2021a/software/bzip2/1.0.8-GCCcore-9.3.0/bin:/software/kp920-RL8/Stages/2021a/software/libxml2/2.9.10-babe7503-GCCcore-9.3.0/bin:/software/kp920-RL8/Stages/2021a/software/XZ/5.2.5-GCCcore-9.3.0/bin:/software/kp920-RL8/Stages/2021a/software/numactl/2.0.14-GCCcore-9.3.0/bin:/software/kp920-RL8/Stages/2021a/software/binutils/2.36.1-GCCcore-9.3.0/bin:/software/kp920-RL8/Stages/2021a/software/GCCcore/9.3.0/bin:/software/libs/arm-optimized-routines/21.02/Kunpeng920/usr/bin:/software/kp920-RL8/Stages/2021a/software/zsh/5.8/bin:/software/kp920-RL8/Stages/2021a/software/tmux/3.2a/bin:/usr/local/bin:/usr/bin:/usr/local/sbin:/usr/sbin
    PWD=/home/snassyr/SourceCode/test/CANN/TensorFlow/ZhongMNIST
    SHELL=/software/kp920-RL8/Stages/2021a/software/zsh/5.8/bin/zsh
    SHLVL=2
    SOFTWAREPLATFORM=kp920-RL8
    SOFTWAREPREFIX=/software
    SOFTWAREROOT=/software/kp920-RL8
    SSH_AUTH_SOCK=/tmp/ssh-q2AMF1cxp2/agent.922263
    SSH_CLIENT=192.168.23.1 48150 22
    SSH_CONNECTION=192.168.23.1 51432 192.168.23.51 22
    SSH_TTY=/dev/pts/0
    TERM=screen
    TERM_PROGRAM=tmux
    TERM_PROGRAM_VERSION=3.2a
    TMUX=/tmp/tmux-1111800007/default,15813,1
    TMUX_PANE=%7
    USER=snassyr
    XDG_DATA_DIRS=/software/kp920-RL8/Stages/2021a/software/PAPI/6.0.0.1-70887df7-GCCcore-9.3.0/share:/software/kp920-RL8/Stages/2021a/software/libpfm/4.11.1-f6500e77-GCCcore-9.3.0/share:/software/kp920-RL8/Stages/2021a/software/ncurses/6.2-GCCcore-9.3.0/share:/software/kp920-RL8/Stages/2021a/software/CMake/3.20.0-GCCcore-9.3.0/share:/software/kp920-RL8/Stages/2021a/software/HDF5/1.12.0-goolf-2021a.9/share:/software/kp920-RL8/Stages/2021a/software/SciPy-Stack/2021a-goolf-2021a.9-Python-3.7.5/share:/software/kp920-RL8/Stages/2021a/software/pybind11/2.6.2-GCCcore-9.3.0-Python-3.7.5/share:/software/kp920-RL8/Stages/2021a/software/Python/3.7.5-GCCcore-9.3.0/share:/software/kp920-RL8/Stages/2021a/software/gRPC/1.37.1-GCCcore-9.3.0/share:/software/kp920-RL8/Stages/2021a/software/c-ares/1.17.1-GCCcore-9.3.0/share:/software/kp920-RL8/Stages/2021a/software/libjpeg-turbo/2.0.6-GCCcore-9.3.0/share:/software/kp920-RL8/Stages/2021a/software/NASM/2.15.05-GCCcore-9.3.0/share:/software/kp920-RL8/Stages/2021a/software/Rust/1.52.1-GCCcore-9.3.0/share:/software/kp920-RL8/Stages/2021a/software/PostgreSQL/13.2-GCCcore-9.3.0/share:/software/kp920-RL8/Stages/2021a/software/libffi/3.3-GCCcore-9.3.0/share:/software/kp920-RL8/Stages/2021a/software/libxslt/1.1.34-GCCcore-9.3.0/share:/software/kp920-RL8/Stages/2021a/software/GMP/6.2.1-GCCcore-9.3.0/share:/software/kp920-RL8/Stages/2021a/software/SQLite/3.35.3-GCCcore-9.3.0/share:/software/kp920-RL8/Stages/2021a/software/Tcl/8.6.11-GCCcore-9.3.0/share:/software/kp920-RL8/Stages/2021a/software/libreadline/8.1-GCCcore-9.3.0/share:/software/kp920-RL8/Stages/2021a/software/FFTW/3.3.9-gompi-2021a.9/share:/software/kp920-RL8/Stages/2021a/software/OpenMPI/4.1.2-GCC-9.3.0/share:/software/kp920-RL8/Stages/2021a/software/UCX/1.11.2-GCCcore-9.3.0/share:/software/kp920-RL8/Stages/2021a/software/hwloc/2.4.1-GCCcore-9.3.0/share:/software/kp920-RL8/Stages/2021a/software/X11/20210331-GCCcore-9.3.0/share:/software/kp920-RL8/Stages/2021a/software/fontconfig/2.13.93-GCCcore-9.3.0/share:/software/kp920-RL8/Stages/2021a/software/util-linux/2.35.1-GCCcore-9.3.0/share:/software/kp920-RL8/Stages/2021a/software/gperf/3.1-GCCcore-9.3.0/share:/software/kp920-RL8/Stages/2021a/software/freetype/2.10.4-GCCcore-9.3.0/share:/software/kp920-RL8/Stages/2021a/software/libpng/1.6.37-GCCcore-9.3.0/share:/software/kp920-RL8/Stages/2021a/software/expat/2.4.2-GCCcore-9.3.0/share:/software/kp920-RL8/Stages/2021a/software/libxml2/2.9.10-babe7503-GCCcore-9.3.0/share:/software/kp920-RL8/Stages/2021a/software/XZ/5.2.5-GCCcore-9.3.0/share:/software/kp920-RL8/Stages/2021a/software/numactl/2.0.14-GCCcore-9.3.0/share:/software/kp920-RL8/Stages/2021a/software/zlib/1.2.11-GCCcore-9.3.0/share:/software/kp920-RL8/Stages/2021a/software/binutils/2.36.1-GCCcore-9.3.0/share:/software/kp920-RL8/Stages/2021a/software/GCCcore/9.3.0/share:/software/kp920-RL8/Stages/2021a/software/zsh/5.8/share:/software/kp920-RL8/Stages/2021a/software/tmux/3.2a/share
    XDG_RUNTIME_DIR=/run/user/1111800007
    XDG_SESSION_ID=20
    _=/usr/bin/env
    _LMFILES_=/software/kp920-RL8/Stages/2021a/UI/Tools/tmux/3.2a.lua:/software/kp920-RL8/Stages/2021a/UI/Tools/zsh/5.8.lua:/software/modules/custom/arm-optimized-routines/21.02.lua:/software/modules/system/Architecture/.Kunpeng920.lua:/software/kp920-RL8/Stages/2021a/UI/Tools/GCCcore/.9.3.0.lua:/software/kp920-RL8/Stages/2021a/modules/all/Compiler/GCCcore/9.3.0/binutils/.2.36.1.lua:/software/kp920-RL8/Stages/2021a/UI/Compilers/GCC/9.3.0.lua:/software/kp920-RL8/Stages/2021a/modules/all/Compiler/GCCcore/9.3.0/zlib/.1.2.11.lua:/software/kp920-RL8/Stages/2021a/modules/all/Compiler/GCCcore/9.3.0/numactl/2.0.14.lua:/software/kp920-RL8/Stages/2021a/modules/all/Compiler/GCCcore/9.3.0/XZ/.5.2.5.lua:/software/kp920-RL8/Stages/2021a/modules/all/Compiler/GCCcore/9.3.0/libxml2/.2.9.10-babe7503.lua:/software/kp920-RL8/Stages/2021a/modules/all/Compiler/GCCcore/9.3.0/bzip2/.1.0.8.lua:/software/kp920-RL8/Stages/2021a/modules/all/Compiler/GCCcore/9.3.0/expat/.2.4.2.lua:/software/kp920-RL8/Stages/2021a/modules/all/Compiler/GCCcore/9.3.0/libpng/.1.6.37.lua:/software/kp920-RL8/Stages/2021a/modules/all/Compiler/GCCcore/9.3.0/freetype/.2.10.4.lua:/software/kp920-RL8/Stages/2021a/modules/all/Compiler/GCCcore/9.3.0/gperf/.3.1.lua:/software/kp920-RL8/Stages/2021a/modules/all/Compiler/GCCcore/9.3.0/util-linux/.2.35.1.lua:/software/kp920-RL8/Stages/2021a/modules/all/Compiler/GCCcore/9.3.0/fontconfig/.2.13.93.lua:/software/kp920-RL8/Stages/2021a/modules/all/Compiler/GCCcore/9.3.0/X11/20210331.lua:/software/kp920-RL8/Stages/2021a/modules/all/Compiler/GCCcore/9.3.0/hwloc/2.4.1.lua:/software/kp920-RL8/Stages/2021a/modules/all/Compiler/GCCcore/9.3.0/UCX/1.11.2.lua:/software/kp920-RL8/Stages/2021a/modules/all/Compiler/mpi/GCC/9.3.0/OpenMPI/4.1.2.lua:/software/kp920-RL8/Stages/2021a/modules/all/Compiler/GCC/9.3.0/OpenBLAS/0.3.19.lua:/software/kp920-RL8/Stages/2021a/modules/all/MPI/GCC/9.3.0/OpenMPI/4.1.2/FFTW/3.3.9.lua:/software/kp920-RL8/Stages/2021a/modules/all/MPI/GCC/9.3.0/OpenMPI/4.1.2/ScaLAPACK/2.1.0-OpenBLAS-0.3.19.lua:/software/kp920-RL8/Stages/2021a/modules/all/Compiler/GCCcore/9.3.0/libreadline/.8.1.lua:/software/kp920-RL8/Stages/2021a/modules/all/Compiler/GCCcore/9.3.0/Tcl/8.6.11.lua:/software/kp920-RL8/Stages/2021a/modules/all/Compiler/GCCcore/9.3.0/SQLite/.3.35.3.lua:/software/kp920-RL8/Stages/2021a/modules/all/Compiler/GCCcore/9.3.0/GMP/6.2.1.lua:/software/kp920-RL8/Stages/2021a/modules/all/Compiler/GCCcore/9.3.0/libxslt/.1.1.34.lua:/software/kp920-RL8/Stages/2021a/modules/all/Compiler/GCCcore/9.3.0/libffi/3.3.lua:/software/kp920-RL8/Stages/2021a/modules/all/Compiler/GCCcore/9.3.0/libyaml/0.2.5.lua:/software/kp920-RL8/Stages/2021a/UI/Tools/Java/11.0.10.lua:/software/kp920-RL8/Stages/2021a/modules/all/Compiler/GCCcore/9.3.0/PostgreSQL/13.2.lua:/software/kp920-RL8/Stages/2021a/modules/all/Compiler/GCCcore/9.3.0/Rust/1.52.1.lua:/software/kp920-RL8/Stages/2021a/modules/all/Compiler/GCCcore/9.3.0/protobuf/.3.15.6.lua:/software/kp920-RL8/Stages/2021a/modules/all/Compiler/GCCcore/9.3.0/gflags/.2.2.2.lua:/software/kp920-RL8/Stages/2021a/modules/all/Compiler/GCCcore/9.3.0/libspatialindex/.1.9.3.lua:/software/kp920-RL8/Stages/2021a/modules/all/Compiler/GCCcore/9.3.0/NASM/.2.15.05.lua:/software/kp920-RL8/Stages/2021a/modules/all/Compiler/GCCcore/9.3.0/libjpeg-turbo/.2.0.6.lua:/software/kp920-RL8/Stages/2021a/modules/all/Compiler/GCCcore/9.3.0/abseil-cpp/20210324.1.lua:/software/kp920-RL8/Stages/2021a/modules/all/Compiler/GCCcore/9.3.0/c-ares/1.17.1.lua:/software/kp920-RL8/Stages/2021a/modules/all/Compiler/GCCcore/9.3.0/re2/2021-04-01.lua:/software/kp920-RL8/Stages/2021a/modules/all/Compiler/GCCcore/9.3.0/gRPC/1.37.1.lua:/software/kp920-RL8/Stages/2021a/modules/all/Compiler/GCCcore/9.3.0/Python/3.7.5.lua:/software/kp920-RL8/Stages/2021a/modules/all/Compiler/GCCcore/9.3.0/pybind11/.2.6.2-Python-3.7.5.lua:/software/kp920-RL8/Stages/2021a/modules/all/MPI/GCC/9.3.0/OpenMPI/4.1.2/SciPy-Stack/2021a-Python-3.7.5.lua:/software/kp920-RL8/Stages/2021a/modules/all/Compiler/GCCcore/9.3.0/protobuf-CANN/3.15.6.lua:/software/kp920-RL8/Stages/2021a/modules/all/Compiler/GCCcore/9.3.0/Szip/.2.1.1.lua:/software/kp920-RL8/Stages/2021a/modules/all/MPI/GCC/9.3.0/OpenMPI/4.1.2/HDF5/1.12.0.lua:/software/kp920-RL8/Stages/2021a/modules/all/MPI/GCC/9.3.0/OpenMPI/4.1.2/h5py/3.5.0-Python-3.7.5.lua:/software/kp920-RL8/Stages/2021a/modules/all/Compiler/GCCcore/9.3.0/nsync-CANN/1.22.0.lua:/software/kp920-RL8/Stages/2021a/modules/all/MPI/GCC/9.3.0/OpenMPI/4.1.2/TensorFlow-CANN-Core/1.15.0-Python-3.7.5.lua:/software/kp920-RL8/Stages/2021a/modules/all/Compiler/GCCcore/9.3.0/CMake/3.20.0.lua:/software/kp920-RL8/Stages/2021a/modules/all/MPI/GCC/9.3.0/OpenMPI/4.1.2/CANN-Toolkit/5.0.2.alpha005-Python-3.7.5.lua:/software/kp920-RL8/Stages/2021a/modules/all/MPI/GCC/9.3.0/OpenMPI/4.1.2/CANN-tfplugin/1.7.0-Python-3.7.5.lua:/software/kp920-RL8/Stages/2021a/modules/all/MPI/GCC/9.3.0/OpenMPI/4.1.2/TensorFlow-CANN/1.15.0-Python-3.7.5.lua:/software/kp920-RL8/Stages/2021a/modules/all/Compiler/GCCcore/9.3.0/ncurses/.6.2.lua:/software/kp920-RL8/Stages/2021a/modules/all/Compiler/GCCcore/9.3.0/libpfm/4.11.1-f6500e77.lua:/software/kp920-RL8/Stages/2021a/modules/all/Compiler/GCCcore/9.3.0/PAPI/6.0.0.1-70887df7.lua
    _ModuleTable001_=X01vZHVsZVRhYmxlXyA9IHsKTVR2ZXJzaW9uID0gMywKY19yZWJ1aWxkVGltZSA9IGZhbHNlLApjX3Nob3J0VGltZSA9IGZhbHNlLApkZXB0aFQgPSB7fSwKZmFtaWx5ID0gewpjb21waWxlciA9ICJHQ0MiLAp9LAptVCA9IHsKQXJjaGl0ZWN0dXJlID0gewpmbiA9ICIvc29mdHdhcmUvbW9kdWxlcy9zeXN0ZW0vQXJjaGl0ZWN0dXJlLy5LdW5wZW5nOTIwLmx1YSIsCmZ1bGxOYW1lID0gIkFyY2hpdGVjdHVyZS8uS3VucGVuZzkyMCIsCmxvYWRPcmRlciA9IDQsCnByb3BUID0gewpsbW9kID0gewpzdGlja3kgPSAxLAp9LAp9LApzdGFja0RlcHRoID0gMCwKc3RhdHVzID0gImFjdGl2ZSIsCnVzZXJOYW1lID0gIkFyY2hpdGVjdHVyZS8uS3VucGVuZzkyMCIsCndWID0gIiprdW5w
    _ModuleTable002_=ZW5nLjAwMDAwMDkyMC4qemZpbmFsIiwKfSwKWyJDQU5OLVRvb2xraXQiXSA9IHsKZm4gPSAiL3NvZnR3YXJlL2twOTIwLVJMOC9TdGFnZXMvMjAyMWEvbW9kdWxlcy9hbGwvTVBJL0dDQy85LjMuMC9PcGVuTVBJLzQuMS4yL0NBTk4tVG9vbGtpdC81LjAuMi5hbHBoYTAwNS1QeXRob24tMy43LjUubHVhIiwKZnVsbE5hbWUgPSAiQ0FOTi1Ub29sa2l0LzUuMC4yLmFscGhhMDA1LVB5dGhvbi0zLjcuNSIsCmxvYWRPcmRlciA9IDU1LApwcm9wVCA9IHt9LApzdGFja0RlcHRoID0gMiwKc3RhdHVzID0gImFjdGl2ZSIsCnVzZXJOYW1lID0gIkNBTk4tVG9vbGtpdC81LjAuMi5hbHBoYTAwNS1QeXRob24tMy43LjUiLAp3ViA9ICIwMDAwMDAwMDUuMDAwMDAwMDAwLjAwMDAwMDAwMi4q
    _ModuleTable003_=YWxwaGEuMDAwMDAwMDA1Lip5dGhvbi4qemZpbmFsLS4wMDAwMDAwMDMuMDAwMDAwMDA3LjAwMDAwMDAwNS4qemZpbmFsIiwKfSwKWyJDQU5OLXRmcGx1Z2luIl0gPSB7CmZuID0gIi9zb2Z0d2FyZS9rcDkyMC1STDgvU3RhZ2VzLzIwMjFhL21vZHVsZXMvYWxsL01QSS9HQ0MvOS4zLjAvT3Blbk1QSS80LjEuMi9DQU5OLXRmcGx1Z2luLzEuNy4wLVB5dGhvbi0zLjcuNS5sdWEiLApmdWxsTmFtZSA9ICJDQU5OLXRmcGx1Z2luLzEuNy4wLVB5dGhvbi0zLjcuNSIsCmxvYWRPcmRlciA9IDU2LApwcm9wVCA9IHt9LApzdGFja0RlcHRoID0gMSwKc3RhdHVzID0gImFjdGl2ZSIsCnVzZXJOYW1lID0gIkNBTk4tdGZwbHVnaW4vMS43LjAtUHl0aG9uLTMuNy41IiwKd1YgPSAiMDAwMDAw
    _ModuleTable004_=MDAxLjAwMDAwMDAwNy4qeXRob24uKnpmaW5hbC0uMDAwMDAwMDAzLjAwMDAwMDAwNy4wMDAwMDAwMDUuKnpmaW5hbCIsCn0sCkNNYWtlID0gewpmbiA9ICIvc29mdHdhcmUva3A5MjAtUkw4L1N0YWdlcy8yMDIxYS9tb2R1bGVzL2FsbC9Db21waWxlci9HQ0Njb3JlLzkuMy4wL0NNYWtlLzMuMjAuMC5sdWEiLApmdWxsTmFtZSA9ICJDTWFrZS8zLjIwLjAiLApsb2FkT3JkZXIgPSA1NCwKcHJvcFQgPSB7fSwKc3RhY2tEZXB0aCA9IDMsCnN0YXR1cyA9ICJhY3RpdmUiLAp1c2VyTmFtZSA9ICJDTWFrZS8zLjIwLjAiLAp3ViA9ICIwMDAwMDAwMDMuMDAwMDAwMDIwLip6ZmluYWwiLAp9LApGRlRXID0gewpmbiA9ICIvc29mdHdhcmUva3A5MjAtUkw4L1N0YWdlcy8yMDIxYS9tb2R1
    _ModuleTable005_=bGVzL2FsbC9NUEkvR0NDLzkuMy4wL09wZW5NUEkvNC4xLjIvRkZUVy8zLjMuOS5sdWEiLApmdWxsTmFtZSA9ICJGRlRXLzMuMy45IiwKbG9hZE9yZGVyID0gMjQsCnByb3BUID0ge30sCnN0YWNrRGVwdGggPSAxLApzdGF0dXMgPSAiYWN0aXZlIiwKdXNlck5hbWUgPSAiRkZUVy8zLjMuOSIsCndWID0gIjAwMDAwMDAwMy4wMDAwMDAwMDMuMDAwMDAwMDA5Lip6ZmluYWwiLAp9LApHQ0MgPSB7CmZuID0gIi9zb2Z0d2FyZS9rcDkyMC1STDgvU3RhZ2VzLzIwMjFhL1VJL0NvbXBpbGVycy9HQ0MvOS4zLjAubHVhIiwKZnVsbE5hbWUgPSAiR0NDLzkuMy4wIiwKbG9hZE9yZGVyID0gNywKcHJvcFQgPSB7fSwKc3RhY2tEZXB0aCA9IDAsCnN0YXR1cyA9ICJhY3RpdmUiLAp1c2VyTmFt
    _ModuleTable_Sz_=48
    __LMOD_REF_COUNT_CMAKE_PREFIX_PATH=/software/kp920-RL8/Stages/2021a/software/PAPI/6.0.0.1-70887df7-GCCcore-9.3.0:1;/software/kp920-RL8/Stages/2021a/software/libpfm/4.11.1-f6500e77-GCCcore-9.3.0:1;/software/kp920-RL8/Stages/2021a/software/ncurses/6.2-GCCcore-9.3.0:1;/software/kp920-RL8/Stages/2021a/software/CANN-tfplugin/1.7.0-goolf-2021a.9-Python-3.7.5:1;/software/kp920-RL8/Stages/2021a/software/CANN-Toolkit/5.0.2.alpha005-goolf-2021a.9-Python-3.7.5:1;/software/kp920-RL8/Stages/2021a/software/CMake/3.20.0-GCCcore-9.3.0:1;/software/kp920-RL8/Stages/2021a/software/TensorFlow-CANN-Core/1.15.0-goolf-2021a.9-Python-3.7.5:1;/software/kp920-RL8/Stages/2021a/software/nsync-CANN/1.22.0-GCCcore-9.3.0:1;/software/kp920-RL8/Stages/2021a/software/h5py/3.5.0-goolf-2021a.9-Python-3.7.5:1;/software/kp920-RL8/Stages/2021a/software/HDF5/1.12.0-goolf-2021a.9:1;/software/kp920-RL8/Stages/2021a/software/Szip/2.1.1-GCCcore-9.3.0:1;/software/kp920-RL8/Stages/2021a/software/protobuf-CANN/3.15.6-GCCcore-9.3.0:1;/software/kp920-RL8/Stages/2021a/software/SciPy-Stack/2021a-goolf-2021a.9-Python-3.7.5:1;/software/kp920-RL8/Stages/2021a/software/pybind11/2.6.2-GCCcore-9.3.0-Python-3.7.5:1;/software/kp920-RL8/Stages/2021a/software/Python/3.7.5-GCCcore-9.3.0:1;/software/kp920-RL8/Stages/2021a/software/gRPC/1.37.1-GCCcore-9.3.0:1;/software/kp920-RL8/Stages/2021a/software/re2/2021-04-01-GCCcore-9.3.0:1;/software/kp920-RL8/Stages/2021a/software/c-ares/1.17.1-GCCcore-9.3.0:1;/software/kp920-RL8/Stages/2021a/software/abseil-cpp/20210324.1-GCCcore-9.3.0:1;/software/kp920-RL8/Stages/2021a/software/libjpeg-turbo/2.0.6-GCCcore-9.3.0:1;/software/kp920-RL8/Stages/2021a/software/NASM/2.15.05-GCCcore-9.3.0:1;/software/kp920-RL8/Stages/2021a/software/libspatialindex/1.9.3-GCCcore-9.3.0:1;/software/kp920-RL8/Stages/2021a/software/gflags/2.2.2-GCCcore-9.3.0:1;/software/kp920-RL8/Stages/2021a/software/protobuf/3.15.6-GCCcore-9.3.0:1;/software/kp920-RL8/Stages/2021a/software/Rust/1.52.1-GCCcore-9.3.0:1;/software/kp920-RL8/Stages/2021a/software/PostgreSQL/13.2-GCCcore-9.3.0:1;/software/kp920-RL8/Stages/2021a/software/Java/11.0.10:1;/software/kp920-RL8/Stages/2021a/software/libyaml/0.2.5-GCCcore-9.3.0:1;/software/kp920-RL8/Stages/2021a/software/libffi/3.3-GCCcore-9.3.0:1;/software/kp920-RL8/Stages/2021a/software/libxslt/1.1.34-GCCcore-9.3.0:1;/software/kp920-RL8/Stages/2021a/software/GMP/6.2.1-GCCcore-9.3.0:1;/software/kp920-RL8/Stages/2021a/software/SQLite/3.35.3-GCCcore-9.3.0:1;/software/kp920-RL8/Stages/2021a/software/Tcl/8.6.11-GCCcore-9.3.0:1;/software/kp920-RL8/Stages/2021a/software/libreadline/8.1-GCCcore-9.3.0:1;/software/kp920-RL8/Stages/2021a/software/ScaLAPACK/2.1.0-gompi-2021a.9-OpenBLAS-0.3.19:1;/software/kp920-RL8/Stages/2021a/software/FFTW/3.3.9-gompi-2021a.9:1;/software/kp920-RL8/Stages/2021a/software/OpenBLAS/0.3.19-GCC-9.3.0:1;/software/kp920-RL8/Stages/2021a/software/OpenMPI/4.1.2-GCC-9.3.0:1;/software/kp920-RL8/Stages/2021a/software/UCX/1.11.2-GCCcore-9.3.0:1;/software/kp920-RL8/Stages/2021a/software/hwloc/2.4.1-GCCcore-9.3.0:1;/software/kp920-RL8/Stages/2021a/software/X11/20210331-GCCcore-9.3.0:1;/software/kp920-RL8/Stages/2021a/software/fontconfig/2.13.93-GCCcore-9.3.0:1;/software/kp920-RL8/Stages/2021a/software/util-linux/2.35.1-GCCcore-9.3.0:1;/software/kp920-RL8/Stages/2021a/software/gperf/3.1-GCCcore-9.3.0:1;/software/kp920-RL8/Stages/2021a/software/freetype/2.10.4-GCCcore-9.3.0:1;/software/kp920-RL8/Stages/2021a/software/libpng/1.6.37-GCCcore-9.3.0:1;/software/kp920-RL8/Stages/2021a/software/expat/2.4.2-GCCcore-9.3.0:1;/software/kp920-RL8/Stages/2021a/software/bzip2/1.0.8-GCCcore-9.3.0:1;/software/kp920-RL8/Stages/2021a/software/libxml2/2.9.10-babe7503-GCCcore-9.3.0:1;/software/kp920-RL8/Stages/2021a/software/XZ/5.2.5-GCCcore-9.3.0:1;/software/kp920-RL8/Stages/2021a/software/numactl/2.0.14-GCCcore-9.3.0:1;/software/kp920-RL8/Stages/2021a/software/zlib/1.2.11-GCCcore-9.3.0:1;/software/kp920-RL8/Stages/2021a/software/binutils/2.36.1-GCCcore-9.3.0:1;/software/kp920-RL8/Stages/2021a/software/GCCcore/9.3.0:1;/software/kp920-RL8/Stages/2021a/software/zsh/5.8:1;/software/kp920-RL8/Stages/2021a/software/tmux/3.2a:1
    __LMOD_REF_COUNT_CPATH=/software/kp920-RL8/Stages/2021a/software/PAPI/6.0.0.1-70887df7-GCCcore-9.3.0/include:1;/software/kp920-RL8/Stages/2021a/software/libpfm/4.11.1-f6500e77-GCCcore-9.3.0/include:1;/software/kp920-RL8/Stages/2021a/software/ncurses/6.2-GCCcore-9.3.0/include:1;/software/kp920-RL8/Stages/2021a/software/CANN-Toolkit/5.0.2.alpha005-goolf-2021a.9-Python-3.7.5/ascend-toolkit/5.0.2.alpha005/acllib/include:1;/software/kp920-RL8/Stages/2021a/software/nsync-CANN/1.22.0-GCCcore-9.3.0/include:1;/software/kp920-RL8/Stages/2021a/software/HDF5/1.12.0-goolf-2021a.9/include:1;/software/kp920-RL8/Stages/2021a/software/Szip/2.1.1-GCCcore-9.3.0/include:1;/software/kp920-RL8/Stages/2021a/software/protobuf-CANN/3.15.6-GCCcore-9.3.0/include:1;/software/kp920-RL8/Stages/2021a/software/SciPy-Stack/2021a-goolf-2021a.9-Python-3.7.5/lib/python3.7/site-packages/numpy/core/include:1;/software/kp920-RL8/Stages/2021a/software/pybind11/2.6.2-GCCcore-9.3.0-Python-3.7.5/include:1;/software/kp920-RL8/Stages/2021a/software/Python/3.7.5-GCCcore-9.3.0/include:1;/software/kp920-RL8/Stages/2021a/software/gRPC/1.37.1-GCCcore-9.3.0/include:1;/software/kp920-RL8/Stages/2021a/software/re2/2021-04-01-GCCcore-9.3.0/include:1;/software/kp920-RL8/Stages/2021a/software/c-ares/1.17.1-GCCcore-9.3.0/include:1;/software/kp920-RL8/Stages/2021a/software/abseil-cpp/20210324.1-GCCcore-9.3.0/include:1;/software/kp920-RL8/Stages/2021a/software/libjpeg-turbo/2.0.6-GCCcore-9.3.0/include:1;/software/kp920-RL8/Stages/2021a/software/libspatialindex/1.9.3-GCCcore-9.3.0/include:1;/software/kp920-RL8/Stages/2021a/software/gflags/2.2.2-GCCcore-9.3.0/include:1;/software/kp920-RL8/Stages/2021a/software/protobuf/3.15.6-GCCcore-9.3.0/include:1;/software/kp920-RL8/Stages/2021a/software/PostgreSQL/13.2-GCCcore-9.3.0/include:1;/software/kp920-RL8/Stages/2021a/software/Java/11.0.10/include:1;/software/kp920-RL8/Stages/2021a/software/libyaml/0.2.5-GCCcore-9.3.0/include:1;/software/kp920-RL8/Stages/2021a/software/libffi/3.3-GCCcore-9.3.0/include:1;/software/kp920-RL8/Stages/2021a/software/libxslt/1.1.34-GCCcore-9.3.0/include:1;/software/kp920-RL8/Stages/2021a/software/GMP/6.2.1-GCCcore-9.3.0/include:1;/software/kp920-RL8/Stages/2021a/software/SQLite/3.35.3-GCCcore-9.3.0/include:1;/software/kp920-RL8/Stages/2021a/software/Tcl/8.6.11-GCCcore-9.3.0/include:1;/software/kp920-RL8/Stages/2021a/software/libreadline/8.1-GCCcore-9.3.0/include:1;/software/kp920-RL8/Stages/2021a/software/ScaLAPACK/2.1.0-gompi-2021a.9-OpenBLAS-0.3.19/include:1;/software/kp920-RL8/Stages/2021a/software/FFTW/3.3.9-gompi-2021a.9/include:1;/software/kp920-RL8/Stages/2021a/software/OpenBLAS/0.3.19-GCC-9.3.0/include:1;/software/kp920-RL8/Stages/2021a/software/OpenMPI/4.1.2-GCC-9.3.0/include:1;/software/kp920-RL8/Stages/2021a/software/UCX/1.11.2-GCCcore-9.3.0/include:1;/software/kp920-RL8/Stages/2021a/software/hwloc/2.4.1-GCCcore-9.3.0/include:1;/software/kp920-RL8/Stages/2021a/software/X11/20210331-GCCcore-9.3.0/include:1;/software/kp920-RL8/Stages/2021a/software/fontconfig/2.13.93-GCCcore-9.3.0/include:1;/software/kp920-RL8/Stages/2021a/software/util-linux/2.35.1-GCCcore-9.3.0/include:1;/software/kp920-RL8/Stages/2021a/software/freetype/2.10.4-GCCcore-9.3.0/include/freetype2:1;/software/kp920-RL8/Stages/2021a/software/libpng/1.6.37-GCCcore-9.3.0/include:1;/software/kp920-RL8/Stages/2021a/software/expat/2.4.2-GCCcore-9.3.0/include:1;/software/kp920-RL8/Stages/2021a/software/bzip2/1.0.8-GCCcore-9.3.0/include:1;/software/kp920-RL8/Stages/2021a/software/libxml2/2.9.10-babe7503-GCCcore-9.3.0/include/libxml2:1;/software/kp920-RL8/Stages/2021a/software/libxml2/2.9.10-babe7503-GCCcore-9.3.0/include:1;/software/kp920-RL8/Stages/2021a/software/XZ/5.2.5-GCCcore-9.3.0/include:1;/software/kp920-RL8/Stages/2021a/software/numactl/2.0.14-GCCcore-9.3.0/include:1;/software/kp920-RL8/Stages/2021a/software/zlib/1.2.11-GCCcore-9.3.0/include:1;/software/kp920-RL8/Stages/2021a/software/binutils/2.36.1-GCCcore-9.3.0/include:1;/software/libs/arm-optimized-routines/21.02/Kunpeng920/usr/include:1
    __LMOD_REF_COUNT_FPATH=/software/kp920-RL8/Stages/2021a/software/zsh/5.8/share/zsh/5.8/functions:1;/software/kp920-RL8/Stages/2021a/software/zsh/5.8/share/zsh/site-functions:1;/opt/ohpc/admin/lmod/lmod/init/ksh_funcs:1
    __LMOD_REF_COUNT_INCLUDE=/software/libs/arm-optimized-routines/21.02/Kunpeng920/usr/include:1
    __LMOD_REF_COUNT_LD_LIBRARY_PATH=/software/kp920-RL8/Stages/2021a/software/PAPI/6.0.0.1-70887df7-GCCcore-9.3.0/lib:1;/software/kp920-RL8/Stages/2021a/software/libpfm/4.11.1-f6500e77-GCCcore-9.3.0/lib:1;/software/kp920-RL8/Stages/2021a/software/ncurses/6.2-GCCcore-9.3.0/lib:1;/software/kp920-RL8/Stages/2021a/software/CANN-Toolkit/5.0.2.alpha005-goolf-2021a.9-Python-3.7.5/ascend-toolkit/5.0.2.alpha005/atc/lib64:1;/software/kp920-RL8/Stages/2021a/software/CANN-Toolkit/5.0.2.alpha005-goolf-2021a.9-Python-3.7.5/ascend-toolkit/5.0.2.alpha005/fwkacllib/lib64:1;/software/kp920-RL8/Stages/2021a/software/nsync-CANN/1.22.0-GCCcore-9.3.0/lib:1;/software/kp920-RL8/Stages/2021a/software/HDF5/1.12.0-goolf-2021a.9/lib:1;/software/kp920-RL8/Stages/2021a/software/Szip/2.1.1-GCCcore-9.3.0/lib:1;/software/kp920-RL8/Stages/2021a/software/protobuf-CANN/3.15.6-GCCcore-9.3.0/lib:1;/software/kp920-RL8/Stages/2021a/software/SciPy-Stack/2021a-goolf-2021a.9-Python-3.7.5/lib/python3.7/site-packages/numpy/core/lib:1;/software/kp920-RL8/Stages/2021a/software/SciPy-Stack/2021a-goolf-2021a.9-Python-3.7.5/lib:1;/software/kp920-RL8/Stages/2021a/software/Python/3.7.5-GCCcore-9.3.0/lib:1;/software/kp920-RL8/Stages/2021a/software/gRPC/1.37.1-GCCcore-9.3.0/lib:1;/software/kp920-RL8/Stages/2021a/software/re2/2021-04-01-GCCcore-9.3.0/lib:1;/software/kp920-RL8/Stages/2021a/software/c-ares/1.17.1-GCCcore-9.3.0/lib:1;/software/kp920-RL8/Stages/2021a/software/abseil-cpp/20210324.1-GCCcore-9.3.0/lib:1;/software/kp920-RL8/Stages/2021a/software/libjpeg-turbo/2.0.6-GCCcore-9.3.0/lib:1;/software/kp920-RL8/Stages/2021a/software/libspatialindex/1.9.3-GCCcore-9.3.0/lib:1;/software/kp920-RL8/Stages/2021a/software/gflags/2.2.2-GCCcore-9.3.0/lib:1;/software/kp920-RL8/Stages/2021a/software/protobuf/3.15.6-GCCcore-9.3.0/lib:1;/software/kp920-RL8/Stages/2021a/software/Rust/1.52.1-GCCcore-9.3.0/lib:1;/software/kp920-RL8/Stages/2021a/software/PostgreSQL/13.2-GCCcore-9.3.0/lib:1;/software/kp920-RL8/Stages/2021a/software/Java/11.0.10/lib:1;/software/kp920-RL8/Stages/2021a/software/libyaml/0.2.5-GCCcore-9.3.0/lib:1;/software/kp920-RL8/Stages/2021a/software/libffi/3.3-GCCcore-9.3.0/lib64:1;/software/kp920-RL8/Stages/2021a/software/libffi/3.3-GCCcore-9.3.0/lib:1;/software/kp920-RL8/Stages/2021a/software/libxslt/1.1.34-GCCcore-9.3.0/lib:1;/software/kp920-RL8/Stages/2021a/software/GMP/6.2.1-GCCcore-9.3.0/lib:1;/software/kp920-RL8/Stages/2021a/software/SQLite/3.35.3-GCCcore-9.3.0/lib:1;/software/kp920-RL8/Stages/2021a/software/Tcl/8.6.11-GCCcore-9.3.0/lib:1;/software/kp920-RL8/Stages/2021a/software/libreadline/8.1-GCCcore-9.3.0/lib:1;/software/kp920-RL8/Stages/2021a/software/ScaLAPACK/2.1.0-gompi-2021a.9-OpenBLAS-0.3.19/lib:1;/software/kp920-RL8/Stages/2021a/software/FFTW/3.3.9-gompi-2021a.9/lib:1;/software/kp920-RL8/Stages/2021a/software/OpenBLAS/0.3.19-GCC-9.3.0/lib:1;/software/kp920-RL8/Stages/2021a/software/OpenMPI/4.1.2-GCC-9.3.0/lib:1;/software/kp920-RL8/Stages/2021a/software/UCX/1.11.2-GCCcore-9.3.0/lib:1;/software/kp920-RL8/Stages/2021a/software/hwloc/2.4.1-GCCcore-9.3.0/lib:1;/software/kp920-RL8/Stages/2021a/software/X11/20210331-GCCcore-9.3.0/lib:1;/software/kp920-RL8/Stages/2021a/software/fontconfig/2.13.93-GCCcore-9.3.0/lib:1;/software/kp920-RL8/Stages/2021a/software/util-linux/2.35.1-GCCcore-9.3.0/lib:1;/software/kp920-RL8/Stages/2021a/software/freetype/2.10.4-GCCcore-9.3.0/lib:1;/software/kp920-RL8/Stages/2021a/software/libpng/1.6.37-GCCcore-9.3.0/lib:1;/software/kp920-RL8/Stages/2021a/software/expat/2.4.2-GCCcore-9.3.0/lib:1;/software/kp920-RL8/Stages/2021a/software/bzip2/1.0.8-GCCcore-9.3.0/lib:1;/software/kp920-RL8/Stages/2021a/software/libxml2/2.9.10-babe7503-GCCcore-9.3.0/lib:1;/software/kp920-RL8/Stages/2021a/software/XZ/5.2.5-GCCcore-9.3.0/lib:1;/software/kp920-RL8/Stages/2021a/software/numactl/2.0.14-GCCcore-9.3.0/lib:1;/software/kp920-RL8/Stages/2021a/software/zlib/1.2.11-GCCcore-9.3.0/lib:1;/software/kp920-RL8/Stages/2021a/software/binutils/2.36.1-GCCcore-9.3.0/lib:1;/software/kp920-RL8/Stages/2021a/software/GCCcore/9.3.0/lib64:1;/software/kp920-RL8/Stages/2021a/software/GCCcore/9.3.0/lib:1;/software/libs/arm-optimized-routines/21.02/Kunpeng920/usr/lib:1;/software/kp920-RL8/Stages/2021a/software/zsh/5.8/lib:1
    __LMOD_REF_COUNT_LIBRARY_PATH=/software/kp920-RL8/Stages/2021a/software/PAPI/6.0.0.1-70887df7-GCCcore-9.3.0/lib:1;/software/kp920-RL8/Stages/2021a/software/libpfm/4.11.1-f6500e77-GCCcore-9.3.0/lib:1;/software/kp920-RL8/Stages/2021a/software/ncurses/6.2-GCCcore-9.3.0/lib:1;/software/kp920-RL8/Stages/2021a/software/CANN-tfplugin/1.7.0-goolf-2021a.9-Python-3.7.5/lib:1;/software/kp920-RL8/Stages/2021a/software/CANN-Toolkit/5.0.2.alpha005-goolf-2021a.9-Python-3.7.5/ascend-toolkit/5.0.2.alpha005/atc/lib64:1;/software/kp920-RL8/Stages/2021a/software/CANN-Toolkit/5.0.2.alpha005-goolf-2021a.9-Python-3.7.5/ascend-toolkit/5.0.2.alpha005/fwkacllib/lib64:1;/software/kp920-RL8/Stages/2021a/software/TensorFlow-CANN-Core/1.15.0-goolf-2021a.9-Python-3.7.5/lib:1;/software/kp920-RL8/Stages/2021a/software/nsync-CANN/1.22.0-GCCcore-9.3.0/lib:1;/software/kp920-RL8/Stages/2021a/software/HDF5/1.12.0-goolf-2021a.9/lib:1;/software/kp920-RL8/Stages/2021a/software/Szip/2.1.1-GCCcore-9.3.0/lib:1;/software/kp920-RL8/Stages/2021a/software/protobuf-CANN/3.15.6-GCCcore-9.3.0/lib:1;/software/kp920-RL8/Stages/2021a/software/SciPy-Stack/2021a-goolf-2021a.9-Python-3.7.5/lib/python3.7/site-packages/numpy/core/lib:1;/software/kp920-RL8/Stages/2021a/software/SciPy-Stack/2021a-goolf-2021a.9-Python-3.7.5/lib:1;/software/kp920-RL8/Stages/2021a/software/Python/3.7.5-GCCcore-9.3.0/lib:1;/software/kp920-RL8/Stages/2021a/software/gRPC/1.37.1-GCCcore-9.3.0/lib:1;/software/kp920-RL8/Stages/2021a/software/re2/2021-04-01-GCCcore-9.3.0/lib:1;/software/kp920-RL8/Stages/2021a/software/c-ares/1.17.1-GCCcore-9.3.0/lib:1;/software/kp920-RL8/Stages/2021a/software/abseil-cpp/20210324.1-GCCcore-9.3.0/lib:1;/software/kp920-RL8/Stages/2021a/software/libjpeg-turbo/2.0.6-GCCcore-9.3.0/lib:1;/software/kp920-RL8/Stages/2021a/software/libspatialindex/1.9.3-GCCcore-9.3.0/lib:1;/software/kp920-RL8/Stages/2021a/software/gflags/2.2.2-GCCcore-9.3.0/lib:1;/software/kp920-RL8/Stages/2021a/software/protobuf/3.15.6-GCCcore-9.3.0/lib:1;/software/kp920-RL8/Stages/2021a/software/Rust/1.52.1-GCCcore-9.3.0/lib:1;/software/kp920-RL8/Stages/2021a/software/PostgreSQL/13.2-GCCcore-9.3.0/lib:1;/software/kp920-RL8/Stages/2021a/software/Java/11.0.10/lib:1;/software/kp920-RL8/Stages/2021a/software/libyaml/0.2.5-GCCcore-9.3.0/lib:1;/software/kp920-RL8/Stages/2021a/software/libffi/3.3-GCCcore-9.3.0/lib64:1;/software/kp920-RL8/Stages/2021a/software/libffi/3.3-GCCcore-9.3.0/lib:1;/software/kp920-RL8/Stages/2021a/software/libxslt/1.1.34-GCCcore-9.3.0/lib:1;/software/kp920-RL8/Stages/2021a/software/GMP/6.2.1-GCCcore-9.3.0/lib:1;/software/kp920-RL8/Stages/2021a/software/SQLite/3.35.3-GCCcore-9.3.0/lib:1;/software/kp920-RL8/Stages/2021a/software/Tcl/8.6.11-GCCcore-9.3.0/lib:1;/software/kp920-RL8/Stages/2021a/software/libreadline/8.1-GCCcore-9.3.0/lib:1;/software/kp920-RL8/Stages/2021a/software/ScaLAPACK/2.1.0-gompi-2021a.9-OpenBLAS-0.3.19/lib:1;/software/kp920-RL8/Stages/2021a/software/FFTW/3.3.9-gompi-2021a.9/lib:1;/software/kp920-RL8/Stages/2021a/software/OpenBLAS/0.3.19-GCC-9.3.0/lib:1;/software/kp920-RL8/Stages/2021a/software/OpenMPI/4.1.2-GCC-9.3.0/lib:1;/software/kp920-RL8/Stages/2021a/software/UCX/1.11.2-GCCcore-9.3.0/lib:1;/software/kp920-RL8/Stages/2021a/software/hwloc/2.4.1-GCCcore-9.3.0/lib:1;/software/kp920-RL8/Stages/2021a/software/X11/20210331-GCCcore-9.3.0/lib:1;/software/kp920-RL8/Stages/2021a/software/fontconfig/2.13.93-GCCcore-9.3.0/lib:1;/software/kp920-RL8/Stages/2021a/software/util-linux/2.35.1-GCCcore-9.3.0/lib:1;/software/kp920-RL8/Stages/2021a/software/freetype/2.10.4-GCCcore-9.3.0/lib:1;/software/kp920-RL8/Stages/2021a/software/libpng/1.6.37-GCCcore-9.3.0/lib:1;/software/kp920-RL8/Stages/2021a/software/expat/2.4.2-GCCcore-9.3.0/lib:1;/software/kp920-RL8/Stages/2021a/software/bzip2/1.0.8-GCCcore-9.3.0/lib:1;/software/kp920-RL8/Stages/2021a/software/libxml2/2.9.10-babe7503-GCCcore-9.3.0/lib:1;/software/kp920-RL8/Stages/2021a/software/XZ/5.2.5-GCCcore-9.3.0/lib:1;/software/kp920-RL8/Stages/2021a/software/numactl/2.0.14-GCCcore-9.3.0/lib:1;/software/kp920-RL8/Stages/2021a/software/zlib/1.2.11-GCCcore-9.3.0/lib:1;/software/kp920-RL8/Stages/2021a/software/binutils/2.36.1-GCCcore-9.3.0/lib:1;/software/libs/arm-optimized-routines/21.02/Kunpeng920/usr/lib:1;/software/kp920-RL8/Stages/2021a/software/zsh/5.8/lib:1
    __LMOD_REF_COUNT_LOADEDMODULES=tmux/3.2a:1;zsh/5.8:1;arm-optimized-routines/21.02:1;Architecture/.Kunpeng920:1;GCCcore/.9.3.0:1;binutils/.2.36.1:1;GCC/9.3.0:1;zlib/.1.2.11:1;numactl/2.0.14:1;XZ/.5.2.5:1;libxml2/.2.9.10-babe7503:1;bzip2/.1.0.8:1;expat/.2.4.2:1;libpng/.1.6.37:1;freetype/.2.10.4:1;gperf/.3.1:1;util-linux/.2.35.1:1;fontconfig/.2.13.93:1;X11/20210331:1;hwloc/2.4.1:1;UCX/1.11.2:1;OpenMPI/4.1.2:1;OpenBLAS/0.3.19:1;FFTW/3.3.9:1;ScaLAPACK/2.1.0-OpenBLAS-0.3.19:1;libreadline/.8.1:1;Tcl/8.6.11:1;SQLite/.3.35.3:1;GMP/6.2.1:1;libxslt/.1.1.34:1;libffi/3.3:1;libyaml/0.2.5:1;Java/11.0.10:1;PostgreSQL/13.2:1;Rust/1.52.1:1;protobuf/.3.15.6:1;gflags/.2.2.2:1;libspatialindex/.1.9.3:1;NASM/.2.15.05:1;libjpeg-turbo/.2.0.6:1;abseil-cpp/20210324.1:1;c-ares/1.17.1:1;re2/2021-04-01:1;gRPC/1.37.1:1;Python/3.7.5:1;pybind11/.2.6.2-Python-3.7.5:1;SciPy-Stack/2021a-Python-3.7.5:1;protobuf-CANN/3.15.6:1;Szip/.2.1.1:1;HDF5/1.12.0:1;h5py/3.5.0-Python-3.7.5:1;nsync-CANN/1.22.0:1;TensorFlow-CANN-Core/1.15.0-Python-3.7.5:1;CMake/3.20.0:1;CANN-Toolkit/5.0.2.alpha005-Python-3.7.5:1;CANN-tfplugin/1.7.0-Python-3.7.5:1;TensorFlow-CANN/1.15.0-Python-3.7.5:1;ncurses/.6.2:1;libpfm/4.11.1-f6500e77:1;PAPI/6.0.0.1-70887df7:1
    __LMOD_REF_COUNT_MANPATH=/software/kp920-RL8/Stages/2021a/software/PAPI/6.0.0.1-70887df7-GCCcore-9.3.0/share/man:1;/software/kp920-RL8/Stages/2021a/software/libpfm/4.11.1-f6500e77-GCCcore-9.3.0/share/man:1;/software/kp920-RL8/Stages/2021a/software/ncurses/6.2-GCCcore-9.3.0/share/man:1;/software/kp920-RL8/Stages/2021a/software/SciPy-Stack/2021a-goolf-2021a.9-Python-3.7.5/share/man:1;/software/kp920-RL8/Stages/2021a/software/Python/3.7.5-GCCcore-9.3.0/share/man:1;/software/kp920-RL8/Stages/2021a/software/Python/3.7.5-GCCcore-9.3.0/man:1;/software/kp920-RL8/Stages/2021a/software/c-ares/1.17.1-GCCcore-9.3.0/share/man:1;/software/kp920-RL8/Stages/2021a/software/libjpeg-turbo/2.0.6-GCCcore-9.3.0/share/man:1;/software/kp920-RL8/Stages/2021a/software/NASM/2.15.05-GCCcore-9.3.0/share/man:1;/software/kp920-RL8/Stages/2021a/software/Rust/1.52.1-GCCcore-9.3.0/share/man:1;/software/kp920-RL8/Stages/2021a/software/Java/11.0.10/man:1;/software/kp920-RL8/Stages/2021a/software/libffi/3.3-GCCcore-9.3.0/share/man:1;/software/kp920-RL8/Stages/2021a/software/libxslt/1.1.34-GCCcore-9.3.0/share/man:1;/software/kp920-RL8/Stages/2021a/software/SQLite/3.35.3-GCCcore-9.3.0/share/man:1;/software/kp920-RL8/Stages/2021a/software/Tcl/8.6.11-GCCcore-9.3.0/share/man:1;/software/kp920-RL8/Stages/2021a/software/Tcl/8.6.11-GCCcore-9.3.0/man:1;/software/kp920-RL8/Stages/2021a/software/libreadline/8.1-GCCcore-9.3.0/share/man:1;/software/kp920-RL8/Stages/2021a/software/FFTW/3.3.9-gompi-2021a.9/share/man:1;/software/kp920-RL8/Stages/2021a/software/OpenMPI/4.1.2-GCC-9.3.0/share/man:1;/software/kp920-RL8/Stages/2021a/software/hwloc/2.4.1-GCCcore-9.3.0/share/man:1;/software/kp920-RL8/Stages/2021a/software/X11/20210331-GCCcore-9.3.0/share/man:1;/software/kp920-RL8/Stages/2021a/software/util-linux/2.35.1-GCCcore-9.3.0/share/man:1;/software/kp920-RL8/Stages/2021a/software/gperf/3.1-GCCcore-9.3.0/share/man:1;/software/kp920-RL8/Stages/2021a/software/freetype/2.10.4-GCCcore-9.3.0/share/man:1;/software/kp920-RL8/Stages/2021a/software/libpng/1.6.37-GCCcore-9.3.0/share/man:1;/software/kp920-RL8/Stages/2021a/software/bzip2/1.0.8-GCCcore-9.3.0/man:1;/software/kp920-RL8/Stages/2021a/software/libxml2/2.9.10-babe7503-GCCcore-9.3.0/share/man:1;/software/kp920-RL8/Stages/2021a/software/XZ/5.2.5-GCCcore-9.3.0/share/man:1;/software/kp920-RL8/Stages/2021a/software/numactl/2.0.14-GCCcore-9.3.0/share/man:1;/software/kp920-RL8/Stages/2021a/software/zlib/1.2.11-GCCcore-9.3.0/share/man:1;/software/kp920-RL8/Stages/2021a/software/binutils/2.36.1-GCCcore-9.3.0/share/man:1;/software/kp920-RL8/Stages/2021a/software/GCCcore/9.3.0/share/man:1;/software/kp920-RL8/Stages/2021a/software/zsh/5.8/share/man:1;/software/kp920-RL8/Stages/2021a/software/tmux/3.2a/share/man:1
    __LMOD_REF_COUNT_MODULEPATH=/software/kp920-RL8/Stages/2021a/modules/all/MPI/GCC/9.3.0/OpenMPI/4.1.2:1;/software/kp920-RL8/Stages/2021a/modules/all/Compiler/mpi/GCC/9.3.0:1;/software/kp920-RL8/Stages/2021a/modules/all/Compiler/GCC/9.3.0:1;/software/kp920-RL8/Stages/2021a/modules/all/Compiler/GCCcore/9.3.0:1;/software/kp920-RL8/Stages/2021a/UI/Compilers:1;/software/kp920-RL8/Stages/2021a/UI/Tools:1;/software/modules/system:1;/software/modules/custom:1
    __LMOD_REF_COUNT_PATH=/software/kp920-RL8/Stages/2021a/software/PAPI/6.0.0.1-70887df7-GCCcore-9.3.0/bin:1;/software/kp920-RL8/Stages/2021a/software/ncurses/6.2-GCCcore-9.3.0/bin:1;/software/kp920-RL8/Stages/2021a/software/CANN-Toolkit/5.0.2.alpha005-goolf-2021a.9-Python-3.7.5:1;/software/kp920-RL8/Stages/2021a/software/CANN-Toolkit/5.0.2.alpha005-goolf-2021a.9-Python-3.7.5/ascend-toolkit/5.0.2.alpha005/atc/bin:1;/software/kp920-RL8/Stages/2021a/software/CANN-Toolkit/5.0.2.alpha005-goolf-2021a.9-Python-3.7.5/ascend-toolkit/5.0.2.alpha005/fwkacllib/bin:1;/software/kp920-RL8/Stages/2021a/software/CANN-Toolkit/5.0.2.alpha005-goolf-2021a.9-Python-3.7.5/ascend-toolkit/5.0.2.alpha005/fwkacllib/ccec_compiler/bin:1;/software/kp920-RL8/Stages/2021a/software/CMake/3.20.0-GCCcore-9.3.0/bin:1;/software/kp920-RL8/Stages/2021a/software/TensorFlow-CANN-Core/1.15.0-goolf-2021a.9-Python-3.7.5/bin:1;/software/kp920-RL8/Stages/2021a/software/HDF5/1.12.0-goolf-2021a.9/bin:1;/software/kp920-RL8/Stages/2021a/software/protobuf-CANN/3.15.6-GCCcore-9.3.0/bin:1;/software/kp920-RL8/Stages/2021a/software/SciPy-Stack/2021a-goolf-2021a.9-Python-3.7.5/bin:1;/software/kp920-RL8/Stages/2021a/software/pybind11/2.6.2-GCCcore-9.3.0-Python-3.7.5/bin:1;/software/kp920-RL8/Stages/2021a/software/Python/3.7.5-GCCcore-9.3.0/bin:1;/software/kp920-RL8/Stages/2021a/software/gRPC/1.37.1-GCCcore-9.3.0/bin:1;/software/kp920-RL8/Stages/2021a/software/c-ares/1.17.1-GCCcore-9.3.0/bin:1;/software/kp920-RL8/Stages/2021a/software/libjpeg-turbo/2.0.6-GCCcore-9.3.0/bin:1;/software/kp920-RL8/Stages/2021a/software/NASM/2.15.05-GCCcore-9.3.0/bin:1;/software/kp920-RL8/Stages/2021a/software/gflags/2.2.2-GCCcore-9.3.0/bin:1;/software/kp920-RL8/Stages/2021a/software/protobuf/3.15.6-GCCcore-9.3.0/bin:1;/software/kp920-RL8/Stages/2021a/software/Rust/1.52.1-GCCcore-9.3.0/bin:1;/software/kp920-RL8/Stages/2021a/software/PostgreSQL/13.2-GCCcore-9.3.0/bin:1;/software/kp920-RL8/Stages/2021a/software/Java/11.0.10:1;/software/kp920-RL8/Stages/2021a/software/Java/11.0.10/bin:1;/software/kp920-RL8/Stages/2021a/software/libxslt/1.1.34-GCCcore-9.3.0/bin:1;/software/kp920-RL8/Stages/2021a/software/SQLite/3.35.3-GCCcore-9.3.0/bin:1;/software/kp920-RL8/Stages/2021a/software/Tcl/8.6.11-GCCcore-9.3.0/bin:1;/software/kp920-RL8/Stages/2021a/software/FFTW/3.3.9-gompi-2021a.9/bin:1;/software/kp920-RL8/Stages/2021a/software/OpenMPI/4.1.2-GCC-9.3.0/bin:1;/software/kp920-RL8/Stages/2021a/software/UCX/1.11.2-GCCcore-9.3.0/bin:1;/software/kp920-RL8/Stages/2021a/software/hwloc/2.4.1-GCCcore-9.3.0/bin:1;/software/kp920-RL8/Stages/2021a/software/X11/20210331-GCCcore-9.3.0/bin:1;/software/kp920-RL8/Stages/2021a/software/fontconfig/2.13.93-GCCcore-9.3.0/bin:1;/software/kp920-RL8/Stages/2021a/software/util-linux/2.35.1-GCCcore-9.3.0/sbin:1;/software/kp920-RL8/Stages/2021a/software/util-linux/2.35.1-GCCcore-9.3.0/bin:1;/software/kp920-RL8/Stages/2021a/software/gperf/3.1-GCCcore-9.3.0/bin:1;/software/kp920-RL8/Stages/2021a/software/freetype/2.10.4-GCCcore-9.3.0/bin:1;/software/kp920-RL8/Stages/2021a/software/libpng/1.6.37-GCCcore-9.3.0/bin:1;/software/kp920-RL8/Stages/2021a/software/expat/2.4.2-GCCcore-9.3.0/bin:1;/software/kp920-RL8/Stages/2021a/software/bzip2/1.0.8-GCCcore-9.3.0/bin:1;/software/kp920-RL8/Stages/2021a/software/libxml2/2.9.10-babe7503-GCCcore-9.3.0/bin:1;/software/kp920-RL8/Stages/2021a/software/XZ/5.2.5-GCCcore-9.3.0/bin:1;/software/kp920-RL8/Stages/2021a/software/numactl/2.0.14-GCCcore-9.3.0/bin:1;/software/kp920-RL8/Stages/2021a/software/binutils/2.36.1-GCCcore-9.3.0/bin:1;/software/kp920-RL8/Stages/2021a/software/GCCcore/9.3.0/bin:1;/software/libs/arm-optimized-routines/21.02/Kunpeng920/usr/bin:1;/software/kp920-RL8/Stages/2021a/software/zsh/5.8/bin:1;/software/kp920-RL8/Stages/2021a/software/tmux/3.2a/bin:1;/usr/local/bin:1;/usr/bin:1;/usr/local/sbin:1;/usr/sbin:1
    __LMOD_REF_COUNT_XDG_DATA_DIRS=/software/kp920-RL8/Stages/2021a/software/PAPI/6.0.0.1-70887df7-GCCcore-9.3.0/share:1;/software/kp920-RL8/Stages/2021a/software/libpfm/4.11.1-f6500e77-GCCcore-9.3.0/share:1;/software/kp920-RL8/Stages/2021a/software/ncurses/6.2-GCCcore-9.3.0/share:1;/software/kp920-RL8/Stages/2021a/software/CMake/3.20.0-GCCcore-9.3.0/share:1;/software/kp920-RL8/Stages/2021a/software/HDF5/1.12.0-goolf-2021a.9/share:1;/software/kp920-RL8/Stages/2021a/software/SciPy-Stack/2021a-goolf-2021a.9-Python-3.7.5/share:1;/software/kp920-RL8/Stages/2021a/software/pybind11/2.6.2-GCCcore-9.3.0-Python-3.7.5/share:1;/software/kp920-RL8/Stages/2021a/software/Python/3.7.5-GCCcore-9.3.0/share:1;/software/kp920-RL8/Stages/2021a/software/gRPC/1.37.1-GCCcore-9.3.0/share:1;/software/kp920-RL8/Stages/2021a/software/c-ares/1.17.1-GCCcore-9.3.0/share:1;/software/kp920-RL8/Stages/2021a/software/libjpeg-turbo/2.0.6-GCCcore-9.3.0/share:1;/software/kp920-RL8/Stages/2021a/software/NASM/2.15.05-GCCcore-9.3.0/share:1;/software/kp920-RL8/Stages/2021a/software/Rust/1.52.1-GCCcore-9.3.0/share:1;/software/kp920-RL8/Stages/2021a/software/PostgreSQL/13.2-GCCcore-9.3.0/share:1;/software/kp920-RL8/Stages/2021a/software/libffi/3.3-GCCcore-9.3.0/share:1;/software/kp920-RL8/Stages/2021a/software/libxslt/1.1.34-GCCcore-9.3.0/share:1;/software/kp920-RL8/Stages/2021a/software/GMP/6.2.1-GCCcore-9.3.0/share:1;/software/kp920-RL8/Stages/2021a/software/SQLite/3.35.3-GCCcore-9.3.0/share:1;/software/kp920-RL8/Stages/2021a/software/Tcl/8.6.11-GCCcore-9.3.0/share:1;/software/kp920-RL8/Stages/2021a/software/libreadline/8.1-GCCcore-9.3.0/share:1;/software/kp920-RL8/Stages/2021a/software/FFTW/3.3.9-gompi-2021a.9/share:1;/software/kp920-RL8/Stages/2021a/software/OpenMPI/4.1.2-GCC-9.3.0/share:1;/software/kp920-RL8/Stages/2021a/software/UCX/1.11.2-GCCcore-9.3.0/share:1;/software/kp920-RL8/Stages/2021a/software/hwloc/2.4.1-GCCcore-9.3.0/share:1;/software/kp920-RL8/Stages/2021a/software/X11/20210331-GCCcore-9.3.0/share:1;/software/kp920-RL8/Stages/2021a/software/fontconfig/2.13.93-GCCcore-9.3.0/share:1;/software/kp920-RL8/Stages/2021a/software/util-linux/2.35.1-GCCcore-9.3.0/share:1;/software/kp920-RL8/Stages/2021a/software/gperf/3.1-GCCcore-9.3.0/share:1;/software/kp920-RL8/Stages/2021a/software/freetype/2.10.4-GCCcore-9.3.0/share:1;/software/kp920-RL8/Stages/2021a/software/libpng/1.6.37-GCCcore-9.3.0/share:1;/software/kp920-RL8/Stages/2021a/software/expat/2.4.2-GCCcore-9.3.0/share:1;/software/kp920-RL8/Stages/2021a/software/libxml2/2.9.10-babe7503-GCCcore-9.3.0/share:1;/software/kp920-RL8/Stages/2021a/software/XZ/5.2.5-GCCcore-9.3.0/share:1;/software/kp920-RL8/Stages/2021a/software/numactl/2.0.14-GCCcore-9.3.0/share:1;/software/kp920-RL8/Stages/2021a/software/zlib/1.2.11-GCCcore-9.3.0/share:1;/software/kp920-RL8/Stages/2021a/software/binutils/2.36.1-GCCcore-9.3.0/share:1;/software/kp920-RL8/Stages/2021a/software/GCCcore/9.3.0/share:1;/software/kp920-RL8/Stages/2021a/software/zsh/5.8/share:1;/software/kp920-RL8/Stages/2021a/software/tmux/3.2a/share:1
    __LMOD_REF_COUNT__LMFILES_=/software/kp920-RL8/Stages/2021a/UI/Tools/tmux/3.2a.lua:1;/software/kp920-RL8/Stages/2021a/UI/Tools/zsh/5.8.lua:1;/software/modules/custom/arm-optimized-routines/21.02.lua:1;/software/modules/system/Architecture/.Kunpeng920.lua:1;/software/kp920-RL8/Stages/2021a/UI/Tools/GCCcore/.9.3.0.lua:1;/software/kp920-RL8/Stages/2021a/modules/all/Compiler/GCCcore/9.3.0/binutils/.2.36.1.lua:1;/software/kp920-RL8/Stages/2021a/UI/Compilers/GCC/9.3.0.lua:1;/software/kp920-RL8/Stages/2021a/modules/all/Compiler/GCCcore/9.3.0/zlib/.1.2.11.lua:1;/software/kp920-RL8/Stages/2021a/modules/all/Compiler/GCCcore/9.3.0/numactl/2.0.14.lua:1;/software/kp920-RL8/Stages/2021a/modules/all/Compiler/GCCcore/9.3.0/XZ/.5.2.5.lua:1;/software/kp920-RL8/Stages/2021a/modules/all/Compiler/GCCcore/9.3.0/libxml2/.2.9.10-babe7503.lua:1;/software/kp920-RL8/Stages/2021a/modules/all/Compiler/GCCcore/9.3.0/bzip2/.1.0.8.lua:1;/software/kp920-RL8/Stages/2021a/modules/all/Compiler/GCCcore/9.3.0/expat/.2.4.2.lua:1;/software/kp920-RL8/Stages/2021a/modules/all/Compiler/GCCcore/9.3.0/libpng/.1.6.37.lua:1;/software/kp920-RL8/Stages/2021a/modules/all/Compiler/GCCcore/9.3.0/freetype/.2.10.4.lua:1;/software/kp920-RL8/Stages/2021a/modules/all/Compiler/GCCcore/9.3.0/gperf/.3.1.lua:1;/software/kp920-RL8/Stages/2021a/modules/all/Compiler/GCCcore/9.3.0/util-linux/.2.35.1.lua:1;/software/kp920-RL8/Stages/2021a/modules/all/Compiler/GCCcore/9.3.0/fontconfig/.2.13.93.lua:1;/software/kp920-RL8/Stages/2021a/modules/all/Compiler/GCCcore/9.3.0/X11/20210331.lua:1;/software/kp920-RL8/Stages/2021a/modules/all/Compiler/GCCcore/9.3.0/hwloc/2.4.1.lua:1;/software/kp920-RL8/Stages/2021a/modules/all/Compiler/GCCcore/9.3.0/UCX/1.11.2.lua:1;/software/kp920-RL8/Stages/2021a/modules/all/Compiler/mpi/GCC/9.3.0/OpenMPI/4.1.2.lua:1;/software/kp920-RL8/Stages/2021a/modules/all/Compiler/GCC/9.3.0/OpenBLAS/0.3.19.lua:1;/software/kp920-RL8/Stages/2021a/modules/all/MPI/GCC/9.3.0/OpenMPI/4.1.2/FFTW/3.3.9.lua:1;/software/kp920-RL8/Stages/2021a/modules/all/MPI/GCC/9.3.0/OpenMPI/4.1.2/ScaLAPACK/2.1.0-OpenBLAS-0.3.19.lua:1;/software/kp920-RL8/Stages/2021a/modules/all/Compiler/GCCcore/9.3.0/libreadline/.8.1.lua:1;/software/kp920-RL8/Stages/2021a/modules/all/Compiler/GCCcore/9.3.0/Tcl/8.6.11.lua:1;/software/kp920-RL8/Stages/2021a/modules/all/Compiler/GCCcore/9.3.0/SQLite/.3.35.3.lua:1;/software/kp920-RL8/Stages/2021a/modules/all/Compiler/GCCcore/9.3.0/GMP/6.2.1.lua:1;/software/kp920-RL8/Stages/2021a/modules/all/Compiler/GCCcore/9.3.0/libxslt/.1.1.34.lua:1;/software/kp920-RL8/Stages/2021a/modules/all/Compiler/GCCcore/9.3.0/libffi/3.3.lua:1;/software/kp920-RL8/Stages/2021a/modules/all/Compiler/GCCcore/9.3.0/libyaml/0.2.5.lua:1;/software/kp920-RL8/Stages/2021a/UI/Tools/Java/11.0.10.lua:1;/software/kp920-RL8/Stages/2021a/modules/all/Compiler/GCCcore/9.3.0/PostgreSQL/13.2.lua:1;/software/kp920-RL8/Stages/2021a/modules/all/Compiler/GCCcore/9.3.0/Rust/1.52.1.lua:1;/software/kp920-RL8/Stages/2021a/modules/all/Compiler/GCCcore/9.3.0/protobuf/.3.15.6.lua:1;/software/kp920-RL8/Stages/2021a/modules/all/Compiler/GCCcore/9.3.0/gflags/.2.2.2.lua:1;/software/kp920-RL8/Stages/2021a/modules/all/Compiler/GCCcore/9.3.0/libspatialindex/.1.9.3.lua:1;/software/kp920-RL8/Stages/2021a/modules/all/Compiler/GCCcore/9.3.0/NASM/.2.15.05.lua:1;/software/kp920-RL8/Stages/2021a/modules/all/Compiler/GCCcore/9.3.0/libjpeg-turbo/.2.0.6.lua:1;/software/kp920-RL8/Stages/2021a/modules/all/Compiler/GCCcore/9.3.0/abseil-cpp/20210324.1.lua:1;/software/kp920-RL8/Stages/2021a/modules/all/Compiler/GCCcore/9.3.0/c-ares/1.17.1.lua:1;/software/kp920-RL8/Stages/2021a/modules/all/Compiler/GCCcore/9.3.0/re2/2021-04-01.lua:1;/software/kp920-RL8/Stages/2021a/modules/all/Compiler/GCCcore/9.3.0/gRPC/1.37.1.lua:1;/software/kp920-RL8/Stages/2021a/modules/all/Compiler/GCCcore/9.3.0/Python/3.7.5.lua:1;/software/kp920-RL8/Stages/2021a/modules/all/Compiler/GCCcore/9.3.0/pybind11/.2.6.2-Python-3.7.5.lua:1;/software/kp920-RL8/Stages/2021a/modules/all/MPI/GCC/9.3.0/OpenMPI/4.1.2/SciPy-Stack/2021a-Python-3.7.5.lua:1;/software/kp920-RL8/Stages/2021a/modules/all/Compiler/GCCcore/9.3.0/protobuf-CANN/3.15.6.lua:1;/software/kp920-RL8/Stages/2021a/modules/all/Compiler/GCCcore/9.3.0/Szip/.2.1.1.lua:1;/software/kp920-RL8/Stages/2021a/modules/all/MPI/GCC/9.3.0/OpenMPI/4.1.2/HDF5/1.12.0.lua:1;/software/kp920-RL8/Stages/2021a/modules/all/MPI/GCC/9.3.0/OpenMPI/4.1.2/h5py/3.5.0-Python-3.7.5.lua:1;/software/kp920-RL8/Stages/2021a/modules/all/Compiler/GCCcore/9.3.0/nsync-CANN/1.22.0.lua:1;/software/kp920-RL8/Stages/2021a/modules/all/MPI/GCC/9.3.0/OpenMPI/4.1.2/TensorFlow-CANN-Core/1.15.0-Python-3.7.5.lua:1;/software/kp920-RL8/Stages/2021a/modules/all/Compiler/GCCcore/9.3.0/CMake/3.20.0.lua:1;/software/kp920-RL8/Stages/2021a/modules/all/MPI/GCC/9.3.0/OpenMPI/4.1.2/CANN-Toolkit/5.0.2.alpha005-Python-3.7.5.lua:1;/software/kp920-RL8/Stages/2021a/modules/all/MPI/GCC/9.3.0/OpenMPI/4.1.2/CANN-tfplugin/1.7.0-Python-3.7.5.lua:1;/software/kp920-RL8/Stages/2021a/modules/all/MPI/GCC/9.3.0/OpenMPI/4.1.2/TensorFlow-CANN/1.15.0-Python-3.7.5.lua:1;/software/kp920-RL8/Stages/2021a/modules/all/Compiler/GCCcore/9.3.0/ncurses/.6.2.lua:1;/software/kp920-RL8/Stages/2021a/modules/all/Compiler/GCCcore/9.3.0/libpfm/4.11.1-f6500e77.lua:1;/software/kp920-RL8/Stages/2021a/modules/all/Compiler/GCCcore/9.3.0/PAPI/6.0.0.1-70887df7.lua:1
    __LMOD_SET_FPATH=1
    which_declare=typeset -f
    OLDPWD=/home/snassyr
    ZSH=/home/snassyr/.oh-my-zsh
    PAGER=less
    LESS=-R
    LSCOLORS=Gxfxcxdxbxegedabagacad
    INCLUDE=/software/libs/arm-optimized-routines/21.02/Kunpeng920/usr/include
    __LMOD_REF_COUNT_ACLOCAL_PATH=/software/kp920-RL8/Stages/2021a/software/CMake/3.20.0-GCCcore-9.3.0/share/aclocal:1;/software/kp920-RL8/Stages/2021a/software/libxslt/1.1.34-GCCcore-9.3.0/share/aclocal:1;/software/kp920-RL8/Stages/2021a/software/X11/20210331-GCCcore-9.3.0/share/aclocal:1;/software/kp920-RL8/Stages/2021a/software/freetype/2.10.4-GCCcore-9.3.0/share/aclocal:1;/software/kp920-RL8/Stages/2021a/software/libxml2/2.9.10-babe7503-GCCcore-9.3.0/share/aclocal:1
    ACLOCAL_PATH=/software/kp920-RL8/Stages/2021a/software/CMake/3.20.0-GCCcore-9.3.0/share/aclocal:/software/kp920-RL8/Stages/2021a/software/libxslt/1.1.34-GCCcore-9.3.0/share/aclocal:/software/kp920-RL8/Stages/2021a/software/X11/20210331-GCCcore-9.3.0/share/aclocal:/software/kp920-RL8/Stages/2021a/software/freetype/2.10.4-GCCcore-9.3.0/share/aclocal:/software/kp920-RL8/Stages/2021a/software/libxml2/2.9.10-babe7503-GCCcore-9.3.0/share/aclocal
    ASCEND_AICPU_PATH=/software/kp920-RL8/Stages/2021a/software/CANN-Toolkit/5.0.2.alpha005-goolf-2021a.9-Python-3.7.5/ascend-toolkit/5.0.2.alpha005
    ASCEND_OPP_PATH=/software/kp920-RL8/Stages/2021a/software/CANN-Toolkit/5.0.2.alpha005-goolf-2021a.9-Python-3.7.5/ascend-toolkit/5.0.2.alpha005/opp
    __LMOD_REF_COUNT_CMAKE_LIBRARY_PATH=/software/kp920-RL8/Stages/2021a/software/libffi/3.3-GCCcore-9.3.0/lib64:1;/software/kp920-RL8/Stages/2021a/software/GCCcore/9.3.0/lib64:1
    CMAKE_LIBRARY_PATH=/software/kp920-RL8/Stages/2021a/software/libffi/3.3-GCCcore-9.3.0/lib64:/software/kp920-RL8/Stages/2021a/software/GCCcore/9.3.0/lib64
    EBDEVELABSEILMINCPP=/software/kp920-RL8/Stages/2021a/software/abseil-cpp/20210324.1-GCCcore-9.3.0/easybuild/Compiler-GCCcore-9.3.0-abseil-cpp-20210324.1-easybuild-devel
    EBDEVELBINUTILS=/software/kp920-RL8/Stages/2021a/software/binutils/2.36.1-GCCcore-9.3.0/easybuild/Compiler-GCCcore-9.3.0-binutils-.2.36.1-easybuild-devel
    EBDEVELBZIP2=/software/kp920-RL8/Stages/2021a/software/bzip2/1.0.8-GCCcore-9.3.0/easybuild/Compiler-GCCcore-9.3.0-bzip2-.1.0.8-easybuild-devel
    EBDEVELCANNMINTFPLUGIN=/software/kp920-RL8/Stages/2021a/software/CANN-tfplugin/1.7.0-goolf-2021a.9-Python-3.7.5/easybuild/MPI-GCC-9.3.0-OpenMPI-4.1.2-CANN-tfplugin-1.7.0-Python-3.7.5-easybuild-devel
    EBDEVELCANNMINTOOLKIT=/software/kp920-RL8/Stages/2021a/software/CANN-Toolkit/5.0.2.alpha005-goolf-2021a.9-Python-3.7.5/easybuild/MPI-GCC-9.3.0-OpenMPI-4.1.2-CANN-Toolkit-5.0.2.alpha005-Python-3.7.5-easybuild-devel
    EBDEVELCMAKE=/software/kp920-RL8/Stages/2021a/software/CMake/3.20.0-GCCcore-9.3.0/easybuild/Compiler-GCCcore-9.3.0-CMake-3.20.0-easybuild-devel
    EBDEVELCMINARES=/software/kp920-RL8/Stages/2021a/software/c-ares/1.17.1-GCCcore-9.3.0/easybuild/Compiler-GCCcore-9.3.0-c-ares-1.17.1-easybuild-devel
    EBDEVELEXPAT=/software/kp920-RL8/Stages/2021a/software/expat/2.4.2-GCCcore-9.3.0/easybuild/Compiler-GCCcore-9.3.0-expat-.2.4.2-easybuild-devel
    EBDEVELFFTW=/software/kp920-RL8/Stages/2021a/software/FFTW/3.3.9-gompi-2021a.9/easybuild/MPI-GCC-9.3.0-OpenMPI-4.1.2-FFTW-3.3.9-easybuild-devel
    EBDEVELFONTCONFIG=/software/kp920-RL8/Stages/2021a/software/fontconfig/2.13.93-GCCcore-9.3.0/easybuild/Compiler-GCCcore-9.3.0-fontconfig-.2.13.93-easybuild-devel
    EBDEVELFREETYPE=/software/kp920-RL8/Stages/2021a/software/freetype/2.10.4-GCCcore-9.3.0/easybuild/Compiler-GCCcore-9.3.0-freetype-.2.10.4-easybuild-devel
    EBDEVELGCC=/software/kp920-RL8/Stages/2021a/software/GCC/9.3.0/easybuild/Core-GCC-9.3.0-easybuild-devel
    EBDEVELGCCCORE=/software/kp920-RL8/Stages/2021a/software/GCCcore/9.3.0/easybuild/Core-GCCcore-.9.3.0-easybuild-devel
    EBDEVELGFLAGS=/software/kp920-RL8/Stages/2021a/software/gflags/2.2.2-GCCcore-9.3.0/easybuild/Compiler-GCCcore-9.3.0-gflags-.2.2.2-easybuild-devel
    EBDEVELGMP=/software/kp920-RL8/Stages/2021a/software/GMP/6.2.1-GCCcore-9.3.0/easybuild/Compiler-GCCcore-9.3.0-GMP-6.2.1-easybuild-devel
    EBDEVELGPERF=/software/kp920-RL8/Stages/2021a/software/gperf/3.1-GCCcore-9.3.0/easybuild/Compiler-GCCcore-9.3.0-gperf-.3.1-easybuild-devel
    EBDEVELGRPC=/software/kp920-RL8/Stages/2021a/software/gRPC/1.37.1-GCCcore-9.3.0/easybuild/Compiler-GCCcore-9.3.0-gRPC-1.37.1-easybuild-devel
    EBDEVELH5PY=/software/kp920-RL8/Stages/2021a/software/h5py/3.5.0-goolf-2021a.9-Python-3.7.5/easybuild/MPI-GCC-9.3.0-OpenMPI-4.1.2-h5py-3.5.0-Python-3.7.5-easybuild-devel
    EBDEVELHDF5=/software/kp920-RL8/Stages/2021a/software/HDF5/1.12.0-goolf-2021a.9/easybuild/MPI-GCC-9.3.0-OpenMPI-4.1.2-HDF5-1.12.0-easybuild-devel
    EBDEVELHWLOC=/software/kp920-RL8/Stages/2021a/software/hwloc/2.4.1-GCCcore-9.3.0/easybuild/Compiler-GCCcore-9.3.0-hwloc-2.4.1-easybuild-devel
    EBDEVELJAVA=/software/kp920-RL8/Stages/2021a/software/Java/11.0.10/easybuild/Core-Java-11.0.10-easybuild-devel
    EBDEVELLIBFFI=/software/kp920-RL8/Stages/2021a/software/libffi/3.3-GCCcore-9.3.0/easybuild/Compiler-GCCcore-9.3.0-libffi-3.3-easybuild-devel
    EBDEVELLIBJPEGMINTURBO=/software/kp920-RL8/Stages/2021a/software/libjpeg-turbo/2.0.6-GCCcore-9.3.0/easybuild/Compiler-GCCcore-9.3.0-libjpeg-turbo-.2.0.6-easybuild-devel
    EBDEVELLIBPNG=/software/kp920-RL8/Stages/2021a/software/libpng/1.6.37-GCCcore-9.3.0/easybuild/Compiler-GCCcore-9.3.0-libpng-.1.6.37-easybuild-devel
    EBDEVELLIBREADLINE=/software/kp920-RL8/Stages/2021a/software/libreadline/8.1-GCCcore-9.3.0/easybuild/Compiler-GCCcore-9.3.0-libreadline-.8.1-easybuild-devel
    EBDEVELLIBSPATIALINDEX=/software/kp920-RL8/Stages/2021a/software/libspatialindex/1.9.3-GCCcore-9.3.0/easybuild/Compiler-GCCcore-9.3.0-libspatialindex-.1.9.3-easybuild-devel
    EBDEVELLIBXML2=/software/kp920-RL8/Stages/2021a/software/libxml2/2.9.10-babe7503-GCCcore-9.3.0/easybuild/Compiler-GCCcore-9.3.0-libxml2-.2.9.10-babe7503-easybuild-devel
    EBDEVELLIBXSLT=/software/kp920-RL8/Stages/2021a/software/libxslt/1.1.34-GCCcore-9.3.0/easybuild/Compiler-GCCcore-9.3.0-libxslt-.1.1.34-easybuild-devel
    EBDEVELLIBYAML=/software/kp920-RL8/Stages/2021a/software/libyaml/0.2.5-GCCcore-9.3.0/easybuild/Compiler-GCCcore-9.3.0-libyaml-0.2.5-easybuild-devel
    EBDEVELNASM=/software/kp920-RL8/Stages/2021a/software/NASM/2.15.05-GCCcore-9.3.0/easybuild/Compiler-GCCcore-9.3.0-NASM-.2.15.05-easybuild-devel
    EBDEVELNSYNCMINCANN=/software/kp920-RL8/Stages/2021a/software/nsync-CANN/1.22.0-GCCcore-9.3.0/easybuild/Compiler-GCCcore-9.3.0-nsync-CANN-1.22.0-easybuild-devel
    EBDEVELNUMACTL=/software/kp920-RL8/Stages/2021a/software/numactl/2.0.14-GCCcore-9.3.0/easybuild/Compiler-GCCcore-9.3.0-numactl-2.0.14-easybuild-devel
    EBDEVELOPENBLAS=/software/kp920-RL8/Stages/2021a/software/OpenBLAS/0.3.19-GCC-9.3.0/easybuild/Compiler-GCC-9.3.0-OpenBLAS-0.3.19-easybuild-devel
    EBDEVELOPENMPI=/software/kp920-RL8/Stages/2021a/software/OpenMPI/4.1.2-GCC-9.3.0/easybuild/Compiler-mpi-GCC-9.3.0-OpenMPI-4.1.2-easybuild-devel
    EBDEVELPOSTGRESQL=/software/kp920-RL8/Stages/2021a/software/PostgreSQL/13.2-GCCcore-9.3.0/easybuild/Compiler-GCCcore-9.3.0-PostgreSQL-13.2-easybuild-devel
    EBDEVELPROTOBUF=/software/kp920-RL8/Stages/2021a/software/protobuf/3.15.6-GCCcore-9.3.0/easybuild/Compiler-GCCcore-9.3.0-protobuf-.3.15.6-easybuild-devel
    EBDEVELPROTOBUFMINCANN=/software/kp920-RL8/Stages/2021a/software/protobuf-CANN/3.15.6-GCCcore-9.3.0/easybuild/Compiler-GCCcore-9.3.0-protobuf-CANN-3.15.6-easybuild-devel
    EBDEVELPYBIND11=/software/kp920-RL8/Stages/2021a/software/pybind11/2.6.2-GCCcore-9.3.0-Python-3.7.5/easybuild/Compiler-GCCcore-9.3.0-pybind11-.2.6.2-Python-3.7.5-easybuild-devel
    EBDEVELPYTHON=/software/kp920-RL8/Stages/2021a/software/Python/3.7.5-GCCcore-9.3.0/easybuild/Compiler-GCCcore-9.3.0-Python-3.7.5-easybuild-devel
    EBDEVELRE2=/software/kp920-RL8/Stages/2021a/software/re2/2021-04-01-GCCcore-9.3.0/easybuild/Compiler-GCCcore-9.3.0-re2-2021-04-01-easybuild-devel
    EBDEVELRUST=/software/kp920-RL8/Stages/2021a/software/Rust/1.52.1-GCCcore-9.3.0/easybuild/Compiler-GCCcore-9.3.0-Rust-1.52.1-easybuild-devel
    EBDEVELSCALAPACK=/software/kp920-RL8/Stages/2021a/software/ScaLAPACK/2.1.0-gompi-2021a.9-OpenBLAS-0.3.19/easybuild/MPI-GCC-9.3.0-OpenMPI-4.1.2-ScaLAPACK-2.1.0-OpenBLAS-0.3.19-easybuild-devel
    EBDEVELSCIPYMINSTACK=/software/kp920-RL8/Stages/2021a/software/SciPy-Stack/2021a-goolf-2021a.9-Python-3.7.5/easybuild/MPI-GCC-9.3.0-OpenMPI-4.1.2-SciPy-Stack-2021a-Python-3.7.5-easybuild-devel
    EBDEVELSQLITE=/software/kp920-RL8/Stages/2021a/software/SQLite/3.35.3-GCCcore-9.3.0/easybuild/Compiler-GCCcore-9.3.0-SQLite-.3.35.3-easybuild-devel
    EBDEVELSZIP=/software/kp920-RL8/Stages/2021a/software/Szip/2.1.1-GCCcore-9.3.0/easybuild/Compiler-GCCcore-9.3.0-Szip-.2.1.1-easybuild-devel
    EBDEVELTCL=/software/kp920-RL8/Stages/2021a/software/Tcl/8.6.11-GCCcore-9.3.0/easybuild/Compiler-GCCcore-9.3.0-Tcl-8.6.11-easybuild-devel
    EBDEVELTENSORFLOWMINCANN=/software/kp920-RL8/Stages/2021a/software/TensorFlow-CANN/1.15.0-goolf-2021a.9-Python-3.7.5/easybuild/MPI-GCC-9.3.0-OpenMPI-4.1.2-TensorFlow-CANN-1.15.0-Python-3.7.5-easybuild-devel
    EBDEVELTENSORFLOWMINCANNMINCORE=/software/kp920-RL8/Stages/2021a/software/TensorFlow-CANN-Core/1.15.0-goolf-2021a.9-Python-3.7.5/easybuild/MPI-GCC-9.3.0-OpenMPI-4.1.2-TensorFlow-CANN-Core-1.15.0-Python-3.7.5-easybuild-devel
    EBDEVELUCX=/software/kp920-RL8/Stages/2021a/software/UCX/1.11.2-GCCcore-9.3.0/easybuild/Compiler-GCCcore-9.3.0-UCX-1.11.2-easybuild-devel
    EBDEVELUTILMINLINUX=/software/kp920-RL8/Stages/2021a/software/util-linux/2.35.1-GCCcore-9.3.0/easybuild/Compiler-GCCcore-9.3.0-util-linux-.2.35.1-easybuild-devel
    EBDEVELX11=/software/kp920-RL8/Stages/2021a/software/X11/20210331-GCCcore-9.3.0/easybuild/Compiler-GCCcore-9.3.0-X11-20210331-easybuild-devel
    EBDEVELXZ=/software/kp920-RL8/Stages/2021a/software/XZ/5.2.5-GCCcore-9.3.0/easybuild/Compiler-GCCcore-9.3.0-XZ-.5.2.5-easybuild-devel
    EBDEVELZLIB=/software/kp920-RL8/Stages/2021a/software/zlib/1.2.11-GCCcore-9.3.0/easybuild/Compiler-GCCcore-9.3.0-zlib-.1.2.11-easybuild-devel
    EBEXTSLISTPYTHON=setuptools-57.4.0,pip-21.3.1,wheel-0.36.2,pyparsing-3.0.6,packaging-20.9,setuptools_scm-6.3.2,flit-core-3.6.0,tomli-2.0.0,attrs-21.4.0,six-1.15.0,semantic_version-2.8.5,iniconfig-1.1.1,toml-0.10.2,setuptools-rust-0.12.1,zipp-3.4.1,appdirs-1.4.4,certifi-2020.12.5,nose-1.3.7,blist-1.3.6,paycheck-1.0.2,argparse-1.4.0,pbr-5.8.1,lockfile-0.12.2,Cython-0.29.23,python-dateutil-2.8.1,decorator-4.4.2,liac-arff-2.5.0,pycrypto-2.6.1,ecdsa-0.16.1,pyasn1-0.4.8,pycparser-2.20,cffi-1.12.3,ipaddress-1.0.23,asn1crypto-1.4.0,idna-3.1,cryptography-3.4.7,PyNaCl-1.4.0,bcrypt-3.2.0,paramiko-2.7.2,netifaces-0.10.9,netaddr-0.8.0,funcsigs-1.0.2,mock-4.0.3,pytz-2021.1,bitstring-3.1.7,lxml-4.6.3,XlsxWriter-1.4.0,Pygments-2.8.1,backports.shutil_get_terminal_size-1.0.0,wcwidth-0.2.5,prompt_toolkit-3.0.18,PyYAML-5.4.1,psycopg2-2.8.6,protobuf-3.19.3,python-gflags-3.1.2,click-8.0.0rc1,itsdangerous-2.0.0rc2,Werkzeug-2.0.0rc4,MarkupSafe-2.0.1,Jinja2-3.0.3,Flask-2.0.0rc1,Mako-1.1.4,py-1.10.0,more-itertools-8.7.0,pluggy-1.0.0.dev0,atomicwrites-1.4.0,scandir-1.10.0,pathlib2-2.3.5,pytest-6.2.3,pytest-runner-5.3.0,ply-3.11,ipython_genutils-0.2.0,traitlets-5.0.5,pickleshare-0.7.5,pexpect-4.8.0,simplegeneric-0.8.1,parso-0.8.2,jedi-0.18.0,backcall-0.2.0,matplotlib-inline-0.1.2,ipython-7.23.0,urllib3-1.26.4,chardet-4.0.0,requests-2.25.1,greenlet-1.0.0,SQLAlchemy-1.4.12,python-editor-1.0.4,alembic-1.5.8,vcversioner-2.16.0.0,pyrsistent-0.17.3,jsonschema-3.2.0,python-oauth2-1.1.1,Rtree-0.9.7,ClusterShell-1.8.3,cloudpickle-1.6.0,Pillow-8.2.0,toolz-0.11.1,xvfbwrapper-0.2.9,traits-6.2.0,webencodings-0.5.1,html5lib-1.1,isodate-0.6.0,rdflib-5.0.0,SPARQLWrapper-1.8.5,networkx-2.5.1,prov-2.0.0,simplejson-3.17.2,configparser-5.0.2,pydot-1.4.2,pydotplus-2.0.2,olefile-0.47.dev4,argcomplete-1.12.3,grako-3.99.9,pytest-forked-1.3.0,apipkg-1.5,execnet-1.8.0,pytest-xdist-2.2.1,TatSu-4.4.0,psutil-5.8.0,docutils-0.17.1,Babel-2.9.1,alabaster-0.7.12,sphinxcontrib-applehelp-1.0.2,sphinxcontrib-devhelp-1.0.2,sphinxcontrib-htmlhelp-1.0.3,sphinxcontrib-jsmath-1.0.1,sphinxcontrib-qthelp-1.0.3,sphinxcontrib-serializinghtml-1.1.4,imagesize-1.2.0,snowballstemmer-2.1.0,Sphinx-4.0.0b2,sphinx-bootstrap-theme-0.7.1,distlib-0.3.1,filelock-3.0.12,importlib_metadata-4.10.1,virtualenv-20.4.4,pytoml-0.1.21,flit-3.2.0,regex-2021.4.4,intreehooks-1.0,crashtest-0.3.1,pylev-1.3.0,pastel-0.2.1,clikit-0.6.2,jeepney-0.6.0,SecretStorage-3.3.1,keyring-23.5.0,keyrings.alt-4.0.2,tomlkit-0.7.0,shellingham-1.4.0,requests-toolbelt-0.9.1,pyrsistent-0.17.3,pkginfo-1.7.0,jsonschema-3.2.0,webencodings-0.5.1,html5lib-1.1,multidict-5.1.0,yarl-1.6.3,async-timeout-3.0.1,typing_extensions-3.7.4.3,aiohttp-3.7.4.post0,dephell-archive-0.1.7,dephell_argparse-0.1.3,dephell_changelogs-0.0.1,dephell_discover-0.2.10,dephell-licenses-0.1.7,dephell_links-0.1.5,dephell_markers-1.0.3,dephell_pythons-0.1.15,dephell_setuptools-0.2.4,dephell_shells-0.1.5,dephell_specifier-0.2.2,dephell_versioning-0.1.2,dephell_venvs-0.1.18,Cerberus-1.3.3,mistune-0.8.4,m2r-0.2.1,ruamel.yaml.clib-0.2.2,ruamel.yaml-0.17.4,yaspin-1.5.0,dephell-0.8.3,cleo-1.0.0a4,cachy-0.3.0,msgpack-1.0.2,CacheControl-0.12.6,ptyprocess-0.7.0,entrypoints-0.3,poetry-core-1.1.0a6,poetry-1.2.0a2,cached-property-1.5.2,grpcio-1.43.0,grpcio-tools-1.43.0
    EBEXTSLISTSCIPYMINSTACK=Cycler-0.10.0,mpmath-1.2.1,hypothesis-6.12.0,sortedcontainers-2.3.0,numpy-1.19.5,scipy-1.6.3,sympy-1.4,pandas-1.2.4,kiwisolver-1.3.1,matplotlib-3.4.1,xarray-0.17.0,seaborn-0.11.1
    EBEXTSLISTTENSORFLOWMINCANNMINCORE=Keras-Preprocessing-1.1.0,absl-py-0.8.1,opt-einsum-3.1.0,gast-0.2.2,tensorflow-estimator-1.15.1,google-pasta-0.1.8,wrapt-1.11.2,Markdown-3.1.1,tensorboard-1.15.0,Keras-Applications-1.0.8,termcolor-1.1.0,astor-0.8.0,TensorFlow-1.15.0
    EBROOTABSEILMINCPP=/software/kp920-RL8/Stages/2021a/software/abseil-cpp/20210324.1-GCCcore-9.3.0
    EBROOTBINUTILS=/software/kp920-RL8/Stages/2021a/software/binutils/2.36.1-GCCcore-9.3.0
    EBROOTBZIP2=/software/kp920-RL8/Stages/2021a/software/bzip2/1.0.8-GCCcore-9.3.0
    EBROOTCANNMINTFPLUGIN=/software/kp920-RL8/Stages/2021a/software/CANN-tfplugin/1.7.0-goolf-2021a.9-Python-3.7.5
    EBROOTCANNMINTOOLKIT=/software/kp920-RL8/Stages/2021a/software/CANN-Toolkit/5.0.2.alpha005-goolf-2021a.9-Python-3.7.5
    EBROOTCMAKE=/software/kp920-RL8/Stages/2021a/software/CMake/3.20.0-GCCcore-9.3.0
    EBROOTCMINARES=/software/kp920-RL8/Stages/2021a/software/c-ares/1.17.1-GCCcore-9.3.0
    EBROOTEXPAT=/software/kp920-RL8/Stages/2021a/software/expat/2.4.2-GCCcore-9.3.0
    EBROOTFFTW=/software/kp920-RL8/Stages/2021a/software/FFTW/3.3.9-gompi-2021a.9
    EBROOTFONTCONFIG=/software/kp920-RL8/Stages/2021a/software/fontconfig/2.13.93-GCCcore-9.3.0
    EBROOTFREETYPE=/software/kp920-RL8/Stages/2021a/software/freetype/2.10.4-GCCcore-9.3.0
    EBROOTGCC=/software/kp920-RL8/Stages/2021a/software/GCCcore/9.3.0
    EBROOTGCCCORE=/software/kp920-RL8/Stages/2021a/software/GCCcore/9.3.0
    EBROOTGFLAGS=/software/kp920-RL8/Stages/2021a/software/gflags/2.2.2-GCCcore-9.3.0
    EBROOTGMP=/software/kp920-RL8/Stages/2021a/software/GMP/6.2.1-GCCcore-9.3.0
    EBROOTGPERF=/software/kp920-RL8/Stages/2021a/software/gperf/3.1-GCCcore-9.3.0
    EBROOTGRPC=/software/kp920-RL8/Stages/2021a/software/gRPC/1.37.1-GCCcore-9.3.0
    EBROOTH5PY=/software/kp920-RL8/Stages/2021a/software/h5py/3.5.0-goolf-2021a.9-Python-3.7.5
    EBROOTHDF5=/software/kp920-RL8/Stages/2021a/software/HDF5/1.12.0-goolf-2021a.9
    EBROOTHWLOC=/software/kp920-RL8/Stages/2021a/software/hwloc/2.4.1-GCCcore-9.3.0
    EBROOTJAVA=/software/kp920-RL8/Stages/2021a/software/Java/11.0.10
    EBROOTLIBFFI=/software/kp920-RL8/Stages/2021a/software/libffi/3.3-GCCcore-9.3.0
    EBROOTLIBJPEGMINTURBO=/software/kp920-RL8/Stages/2021a/software/libjpeg-turbo/2.0.6-GCCcore-9.3.0
    EBROOTLIBPNG=/software/kp920-RL8/Stages/2021a/software/libpng/1.6.37-GCCcore-9.3.0
    EBROOTLIBREADLINE=/software/kp920-RL8/Stages/2021a/software/libreadline/8.1-GCCcore-9.3.0
    EBROOTLIBSPATIALINDEX=/software/kp920-RL8/Stages/2021a/software/libspatialindex/1.9.3-GCCcore-9.3.0
    EBROOTLIBXML2=/software/kp920-RL8/Stages/2021a/software/libxml2/2.9.10-babe7503-GCCcore-9.3.0
    EBROOTLIBXSLT=/software/kp920-RL8/Stages/2021a/software/libxslt/1.1.34-GCCcore-9.3.0
    EBROOTLIBYAML=/software/kp920-RL8/Stages/2021a/software/libyaml/0.2.5-GCCcore-9.3.0
    EBROOTNASM=/software/kp920-RL8/Stages/2021a/software/NASM/2.15.05-GCCcore-9.3.0
    EBROOTNSYNCMINCANN=/software/kp920-RL8/Stages/2021a/software/nsync-CANN/1.22.0-GCCcore-9.3.0
    EBROOTNUMACTL=/software/kp920-RL8/Stages/2021a/software/numactl/2.0.14-GCCcore-9.3.0
    EBROOTOPENBLAS=/software/kp920-RL8/Stages/2021a/software/OpenBLAS/0.3.19-GCC-9.3.0
    EBROOTOPENMPI=/software/kp920-RL8/Stages/2021a/software/OpenMPI/4.1.2-GCC-9.3.0
    EBROOTPOSTGRESQL=/software/kp920-RL8/Stages/2021a/software/PostgreSQL/13.2-GCCcore-9.3.0
    EBROOTPROTOBUF=/software/kp920-RL8/Stages/2021a/software/protobuf/3.15.6-GCCcore-9.3.0
    EBROOTPROTOBUFMINCANN=/software/kp920-RL8/Stages/2021a/software/protobuf-CANN/3.15.6-GCCcore-9.3.0
    EBROOTPYBIND11=/software/kp920-RL8/Stages/2021a/software/pybind11/2.6.2-GCCcore-9.3.0-Python-3.7.5
    EBROOTPYTHON=/software/kp920-RL8/Stages/2021a/software/Python/3.7.5-GCCcore-9.3.0
    EBROOTRE2=/software/kp920-RL8/Stages/2021a/software/re2/2021-04-01-GCCcore-9.3.0
    EBROOTRUST=/software/kp920-RL8/Stages/2021a/software/Rust/1.52.1-GCCcore-9.3.0
    EBROOTSCALAPACK=/software/kp920-RL8/Stages/2021a/software/ScaLAPACK/2.1.0-gompi-2021a.9-OpenBLAS-0.3.19
    EBROOTSCIPYMINSTACK=/software/kp920-RL8/Stages/2021a/software/SciPy-Stack/2021a-goolf-2021a.9-Python-3.7.5
    EBROOTSQLITE=/software/kp920-RL8/Stages/2021a/software/SQLite/3.35.3-GCCcore-9.3.0
    EBROOTSZIP=/software/kp920-RL8/Stages/2021a/software/Szip/2.1.1-GCCcore-9.3.0
    EBROOTTCL=/software/kp920-RL8/Stages/2021a/software/Tcl/8.6.11-GCCcore-9.3.0
    EBROOTTENSORFLOWMINCANN=/software/kp920-RL8/Stages/2021a/software/TensorFlow-CANN/1.15.0-goolf-2021a.9-Python-3.7.5
    EBROOTTENSORFLOWMINCANNMINCORE=/software/kp920-RL8/Stages/2021a/software/TensorFlow-CANN-Core/1.15.0-goolf-2021a.9-Python-3.7.5
    EBROOTUCX=/software/kp920-RL8/Stages/2021a/software/UCX/1.11.2-GCCcore-9.3.0
    EBROOTUTILMINLINUX=/software/kp920-RL8/Stages/2021a/software/util-linux/2.35.1-GCCcore-9.3.0
    EBROOTX11=/software/kp920-RL8/Stages/2021a/software/X11/20210331-GCCcore-9.3.0
    EBROOTXZ=/software/kp920-RL8/Stages/2021a/software/XZ/5.2.5-GCCcore-9.3.0
    EBROOTZLIB=/software/kp920-RL8/Stages/2021a/software/zlib/1.2.11-GCCcore-9.3.0
    EBVERSIONABSEILMINCPP=20210324.1
    EBVERSIONBINUTILS=2.36.1
    EBVERSIONBZIP2=1.0.8
    EBVERSIONCANNMINTFPLUGIN=1.7.0
    EBVERSIONCANNMINTOOLKIT=5.0.2.alpha005
    EBVERSIONCMAKE=3.20.0
    EBVERSIONCMINARES=1.17.1
    EBVERSIONEXPAT=2.4.2
    EBVERSIONFFTW=3.3.9
    EBVERSIONFONTCONFIG=2.13.93
    EBVERSIONFREETYPE=2.10.4
    EBVERSIONGCC=9.3.0
    EBVERSIONGCCCORE=9.3.0
    EBVERSIONGFLAGS=2.2.2
    EBVERSIONGMP=6.2.1
    EBVERSIONGPERF=3.1
    EBVERSIONGRPC=1.37.1
    EBVERSIONH5PY=3.5.0
    EBVERSIONHDF5=1.12.0
    EBVERSIONHWLOC=2.4.1
    EBVERSIONJAVA=11.0.10
    EBVERSIONLIBFFI=3.3
    EBVERSIONLIBJPEGMINTURBO=2.0.6
    EBVERSIONLIBPNG=1.6.37
    EBVERSIONLIBREADLINE=8.1
    EBVERSIONLIBSPATIALINDEX=1.9.3
    EBVERSIONLIBXML2=2.9.10-babe7503
    EBVERSIONLIBXSLT=1.1.34
    EBVERSIONLIBYAML=0.2.5
    EBVERSIONNASM=2.15.05
    EBVERSIONNSYNCMINCANN=1.22.0
    EBVERSIONNUMACTL=2.0.14
    EBVERSIONOPENBLAS=0.3.19
    EBVERSIONOPENMPI=4.1.2
    EBVERSIONPOSTGRESQL=13.2
    EBVERSIONPROTOBUF=3.15.6
    EBVERSIONPROTOBUFMINCANN=3.15.6
    EBVERSIONPYBIND11=2.6.2
    EBVERSIONPYTHON=3.7.5
    EBVERSIONRE2=2021-04-01
    EBVERSIONRUST=1.52.1
    EBVERSIONSCALAPACK=2.1.0
    EBVERSIONSCIPYMINSTACK=2021a
    EBVERSIONSQLITE=3.35.3
    EBVERSIONSZIP=2.1.1
    EBVERSIONTCL=8.6.11
    EBVERSIONTENSORFLOWMINCANN=1.15.0
    EBVERSIONTENSORFLOWMINCANNMINCORE=1.15.0
    EBVERSIONUCX=1.11.2
    EBVERSIONUTILMINLINUX=2.35.1
    EBVERSIONX11=20210331
    EBVERSIONXZ=5.2.5
    EBVERSIONZLIB=1.2.11
    HDF5_DIR=/software/kp920-RL8/Stages/2021a/software/HDF5/1.12.0-goolf-2021a.9
    JAVA_HOME=/software/kp920-RL8/Stages/2021a/software/Java/11.0.10
    LMOD_FAMILY_COMPILER=GCC
    LMOD_FAMILY_COMPILER_VERSION=9.3.0
    __LMOD_REF_COUNT_PKG_CONFIG_PATH=/software/kp920-RL8/Stages/2021a/software/PAPI/6.0.0.1-70887df7-GCCcore-9.3.0/lib/pkgconfig:1;/software/kp920-RL8/Stages/2021a/software/HDF5/1.12.0-goolf-2021a.9/lib/pkgconfig:1;/software/kp920-RL8/Stages/2021a/software/protobuf-CANN/3.15.6-GCCcore-9.3.0/lib/pkgconfig:1;/software/kp920-RL8/Stages/2021a/software/Python/3.7.5-GCCcore-9.3.0/lib/pkgconfig:1;/software/kp920-RL8/Stages/2021a/software/gRPC/1.37.1-GCCcore-9.3.0/lib/pkgconfig:1;/software/kp920-RL8/Stages/2021a/software/c-ares/1.17.1-GCCcore-9.3.0/lib/pkgconfig:1;/software/kp920-RL8/Stages/2021a/software/abseil-cpp/20210324.1-GCCcore-9.3.0/lib/pkgconfig:1;/software/kp920-RL8/Stages/2021a/software/libjpeg-turbo/2.0.6-GCCcore-9.3.0/lib/pkgconfig:1;/software/kp920-RL8/Stages/2021a/software/gflags/2.2.2-GCCcore-9.3.0/lib/pkgconfig:1;/software/kp920-RL8/Stages/2021a/software/protobuf/3.15.6-GCCcore-9.3.0/lib/pkgconfig:1;/software/kp920-RL8/Stages/2021a/software/PostgreSQL/13.2-GCCcore-9.3.0/lib/pkgconfig:1;/software/kp920-RL8/Stages/2021a/software/libyaml/0.2.5-GCCcore-9.3.0/lib/pkgconfig:1;/software/kp920-RL8/Stages/2021a/software/libffi/3.3-GCCcore-9.3.0/lib/pkgconfig:1;/software/kp920-RL8/Stages/2021a/software/libxslt/1.1.34-GCCcore-9.3.0/lib/pkgconfig:1;/software/kp920-RL8/Stages/2021a/software/GMP/6.2.1-GCCcore-9.3.0/lib/pkgconfig:1;/software/kp920-RL8/Stages/2021a/software/SQLite/3.35.3-GCCcore-9.3.0/lib/pkgconfig:1;/software/kp920-RL8/Stages/2021a/software/Tcl/8.6.11-GCCcore-9.3.0/lib/pkgconfig:1;/software/kp920-RL8/Stages/2021a/software/libreadline/8.1-GCCcore-9.3.0/lib/pkgconfig:1;/software/kp920-RL8/Stages/2021a/software/FFTW/3.3.9-gompi-2021a.9/lib/pkgconfig:1;/software/kp920-RL8/Stages/2021a/software/OpenBLAS/0.3.19-GCC-9.3.0/lib/pkgconfig:1;/software/kp920-RL8/Stages/2021a/software/OpenMPI/4.1.2-GCC-9.3.0/lib/pkgconfig:1;/software/kp920-RL8/Stages/2021a/software/UCX/1.11.2-GCCcore-9.3.0/lib/pkgconfig:1;/software/kp920-RL8/Stages/2021a/software/hwloc/2.4.1-GCCcore-9.3.0/lib/pkgconfig:1;/software/kp920-RL8/Stages/2021a/software/X11/20210331-GCCcore-9.3.0/share/pkgconfig:1;/software/kp920-RL8/Stages/2021a/software/X11/20210331-GCCcore-9.3.0/lib/pkgconfig:1;/software/kp920-RL8/Stages/2021a/software/fontconfig/2.13.93-GCCcore-9.3.0/lib/pkgconfig:1;/software/kp920-RL8/Stages/2021a/software/util-linux/2.35.1-GCCcore-9.3.0/lib/pkgconfig:1;/software/kp920-RL8/Stages/2021a/software/freetype/2.10.4-GCCcore-9.3.0/lib/pkgconfig:1;/software/kp920-RL8/Stages/2021a/software/libpng/1.6.37-GCCcore-9.3.0/lib/pkgconfig:1;/software/kp920-RL8/Stages/2021a/software/expat/2.4.2-GCCcore-9.3.0/lib/pkgconfig:1;/software/kp920-RL8/Stages/2021a/software/bzip2/1.0.8-GCCcore-9.3.0/lib/pkgconfig:1;/software/kp920-RL8/Stages/2021a/software/libxml2/2.9.10-babe7503-GCCcore-9.3.0/lib/pkgconfig:1;/software/kp920-RL8/Stages/2021a/software/XZ/5.2.5-GCCcore-9.3.0/lib/pkgconfig:1;/software/kp920-RL8/Stages/2021a/software/numactl/2.0.14-GCCcore-9.3.0/lib/pkgconfig:1;/software/kp920-RL8/Stages/2021a/software/zlib/1.2.11-GCCcore-9.3.0/lib/pkgconfig:1
    PKG_CONFIG_PATH=/software/kp920-RL8/Stages/2021a/software/PAPI/6.0.0.1-70887df7-GCCcore-9.3.0/lib/pkgconfig:/software/kp920-RL8/Stages/2021a/software/HDF5/1.12.0-goolf-2021a.9/lib/pkgconfig:/software/kp920-RL8/Stages/2021a/software/protobuf-CANN/3.15.6-GCCcore-9.3.0/lib/pkgconfig:/software/kp920-RL8/Stages/2021a/software/Python/3.7.5-GCCcore-9.3.0/lib/pkgconfig:/software/kp920-RL8/Stages/2021a/software/gRPC/1.37.1-GCCcore-9.3.0/lib/pkgconfig:/software/kp920-RL8/Stages/2021a/software/c-ares/1.17.1-GCCcore-9.3.0/lib/pkgconfig:/software/kp920-RL8/Stages/2021a/software/abseil-cpp/20210324.1-GCCcore-9.3.0/lib/pkgconfig:/software/kp920-RL8/Stages/2021a/software/libjpeg-turbo/2.0.6-GCCcore-9.3.0/lib/pkgconfig:/software/kp920-RL8/Stages/2021a/software/gflags/2.2.2-GCCcore-9.3.0/lib/pkgconfig:/software/kp920-RL8/Stages/2021a/software/protobuf/3.15.6-GCCcore-9.3.0/lib/pkgconfig:/software/kp920-RL8/Stages/2021a/software/PostgreSQL/13.2-GCCcore-9.3.0/lib/pkgconfig:/software/kp920-RL8/Stages/2021a/software/libyaml/0.2.5-GCCcore-9.3.0/lib/pkgconfig:/software/kp920-RL8/Stages/2021a/software/libffi/3.3-GCCcore-9.3.0/lib/pkgconfig:/software/kp920-RL8/Stages/2021a/software/libxslt/1.1.34-GCCcore-9.3.0/lib/pkgconfig:/software/kp920-RL8/Stages/2021a/software/GMP/6.2.1-GCCcore-9.3.0/lib/pkgconfig:/software/kp920-RL8/Stages/2021a/software/SQLite/3.35.3-GCCcore-9.3.0/lib/pkgconfig:/software/kp920-RL8/Stages/2021a/software/Tcl/8.6.11-GCCcore-9.3.0/lib/pkgconfig:/software/kp920-RL8/Stages/2021a/software/libreadline/8.1-GCCcore-9.3.0/lib/pkgconfig:/software/kp920-RL8/Stages/2021a/software/FFTW/3.3.9-gompi-2021a.9/lib/pkgconfig:/software/kp920-RL8/Stages/2021a/software/OpenBLAS/0.3.19-GCC-9.3.0/lib/pkgconfig:/software/kp920-RL8/Stages/2021a/software/OpenMPI/4.1.2-GCC-9.3.0/lib/pkgconfig:/software/kp920-RL8/Stages/2021a/software/UCX/1.11.2-GCCcore-9.3.0/lib/pkgconfig:/software/kp920-RL8/Stages/2021a/software/hwloc/2.4.1-GCCcore-9.3.0/lib/pkgconfig:/software/kp920-RL8/Stages/2021a/software/X11/20210331-GCCcore-9.3.0/share/pkgconfig:/software/kp920-RL8/Stages/2021a/software/X11/20210331-GCCcore-9.3.0/lib/pkgconfig:/software/kp920-RL8/Stages/2021a/software/fontconfig/2.13.93-GCCcore-9.3.0/lib/pkgconfig:/software/kp920-RL8/Stages/2021a/software/util-linux/2.35.1-GCCcore-9.3.0/lib/pkgconfig:/software/kp920-RL8/Stages/2021a/software/freetype/2.10.4-GCCcore-9.3.0/lib/pkgconfig:/software/kp920-RL8/Stages/2021a/software/libpng/1.6.37-GCCcore-9.3.0/lib/pkgconfig:/software/kp920-RL8/Stages/2021a/software/expat/2.4.2-GCCcore-9.3.0/lib/pkgconfig:/software/kp920-RL8/Stages/2021a/software/bzip2/1.0.8-GCCcore-9.3.0/lib/pkgconfig:/software/kp920-RL8/Stages/2021a/software/libxml2/2.9.10-babe7503-GCCcore-9.3.0/lib/pkgconfig:/software/kp920-RL8/Stages/2021a/software/XZ/5.2.5-GCCcore-9.3.0/lib/pkgconfig:/software/kp920-RL8/Stages/2021a/software/numactl/2.0.14-GCCcore-9.3.0/lib/pkgconfig:/software/kp920-RL8/Stages/2021a/software/zlib/1.2.11-GCCcore-9.3.0/lib/pkgconfig
    __LMOD_REF_COUNT_PYTHONPATH=/software/kp920-RL8/Stages/2021a/software/CANN-tfplugin/1.7.0-goolf-2021a.9-Python-3.7.5/lib/python/site-packages:1;/software/kp920-RL8/Stages/2021a/software/CANN-Toolkit/5.0.2.alpha005-goolf-2021a.9-Python-3.7.5/ascend-toolkit/5.0.2.alpha005/pyACL/python/site-packages/acl:1;/software/kp920-RL8/Stages/2021a/software/CANN-Toolkit/5.0.2.alpha005-goolf-2021a.9-Python-3.7.5/ascend-toolkit/5.0.2.alpha005/atc/python/site-packages:1;/software/kp920-RL8/Stages/2021a/software/CANN-Toolkit/5.0.2.alpha005-goolf-2021a.9-Python-3.7.5/ascend-toolkit/5.0.2.alpha005/toolkit/python/site-packages:1;/software/kp920-RL8/Stages/2021a/software/CANN-Toolkit/5.0.2.alpha005-goolf-2021a.9-Python-3.7.5/ascend-toolkit/5.0.2.alpha005/fwkacllib/python/site-packages:1;/software/kp920-RL8/Stages/2021a/software/TensorFlow-CANN-Core/1.15.0-goolf-2021a.9-Python-3.7.5/lib/python3.7/site-packages:1;/software/kp920-RL8/Stages/2021a/software/h5py/3.5.0-goolf-2021a.9-Python-3.7.5/lib/python3.7/site-packages:1;/software/kp920-RL8/Stages/2021a/software/SciPy-Stack/2021a-goolf-2021a.9-Python-3.7.5/lib/python3.7/site-packages:1;/software/kp920-RL8/Stages/2021a/software/pybind11/2.6.2-GCCcore-9.3.0-Python-3.7.5/lib/python3.7/site-packages:1;/software/kp920-RL8/Stages/2021a/software/Python/3.7.5-GCCcore-9.3.0/easybuild/python:1
    PYTHONPATH=/software/kp920-RL8/Stages/2021a/software/CANN-tfplugin/1.7.0-goolf-2021a.9-Python-3.7.5/lib/python/site-packages:/software/kp920-RL8/Stages/2021a/software/CANN-Toolkit/5.0.2.alpha005-goolf-2021a.9-Python-3.7.5/ascend-toolkit/5.0.2.alpha005/pyACL/python/site-packages/acl:/software/kp920-RL8/Stages/2021a/software/CANN-Toolkit/5.0.2.alpha005-goolf-2021a.9-Python-3.7.5/ascend-toolkit/5.0.2.alpha005/atc/python/site-packages:/software/kp920-RL8/Stages/2021a/software/CANN-Toolkit/5.0.2.alpha005-goolf-2021a.9-Python-3.7.5/ascend-toolkit/5.0.2.alpha005/toolkit/python/site-packages:/software/kp920-RL8/Stages/2021a/software/CANN-Toolkit/5.0.2.alpha005-goolf-2021a.9-Python-3.7.5/ascend-toolkit/5.0.2.alpha005/fwkacllib/python/site-packages:/software/kp920-RL8/Stages/2021a/software/TensorFlow-CANN-Core/1.15.0-goolf-2021a.9-Python-3.7.5/lib/python3.7/site-packages:/software/kp920-RL8/Stages/2021a/software/h5py/3.5.0-goolf-2021a.9-Python-3.7.5/lib/python3.7/site-packages:/software/kp920-RL8/Stages/2021a/software/SciPy-Stack/2021a-goolf-2021a.9-Python-3.7.5/lib/python3.7/site-packages:/software/kp920-RL8/Stages/2021a/software/pybind11/2.6.2-GCCcore-9.3.0-Python-3.7.5/lib/python3.7/site-packages:/software/kp920-RL8/Stages/2021a/software/Python/3.7.5-GCCcore-9.3.0/easybuild/python
    TOOLCHAIN_HOME=/software/kp920-RL8/Stages/2021a/software/CANN-Toolkit/5.0.2.alpha005-goolf-2021a.9-Python-3.7.5/ascend-toolkit/5.0.2.alpha005/toolkit
    _ModuleTable006_=ZSA9ICJHQ0MvOS4zLjAiLAp3ViA9ICIwMDAwMDAwMDkuMDAwMDAwMDAzLip6ZmluYWwiLAp9LApHQ0Njb3JlID0gewpmbiA9ICIvc29mdHdhcmUva3A5MjAtUkw4L1N0YWdlcy8yMDIxYS9VSS9Ub29scy9HQ0Njb3JlLy45LjMuMC5sdWEiLApmdWxsTmFtZSA9ICJHQ0Njb3JlLy45LjMuMCIsCmxvYWRPcmRlciA9IDUsCnByb3BUID0ge30sCnN0YWNrRGVwdGggPSAxLApzdGF0dXMgPSAiYWN0aXZlIiwKdXNlck5hbWUgPSAiR0NDY29yZS8uOS4zLjAiLAp3ViA9ICIwMDAwMDAwMDAuMDAwMDAwMDA5LjAwMDAwMDAwMy4qemZpbmFsIiwKfSwKR01QID0gewpmbiA9ICIvc29mdHdhcmUva3A5MjAtUkw4L1N0YWdlcy8yMDIxYS9tb2R1bGVzL2FsbC9Db21waWxlci9HQ0Njb3JlLzku
    _ModuleTable007_=My4wL0dNUC82LjIuMS5sdWEiLApmdWxsTmFtZSA9ICJHTVAvNi4yLjEiLApsb2FkT3JkZXIgPSAyOSwKcHJvcFQgPSB7fSwKc3RhY2tEZXB0aCA9IDIsCnN0YXR1cyA9ICJhY3RpdmUiLAp1c2VyTmFtZSA9ICJHTVAvNi4yLjEiLAp3ViA9ICIwMDAwMDAwMDYuMDAwMDAwMDAyLjAwMDAwMDAwMS4qemZpbmFsIiwKfSwKSERGNSA9IHsKZm4gPSAiL3NvZnR3YXJlL2twOTIwLVJMOC9TdGFnZXMvMjAyMWEvbW9kdWxlcy9hbGwvTVBJL0dDQy85LjMuMC9PcGVuTVBJLzQuMS4yL0hERjUvMS4xMi4wLmx1YSIsCmZ1bGxOYW1lID0gIkhERjUvMS4xMi4wIiwKbG9hZE9yZGVyID0gNTAsCnByb3BUID0ge30sCnN0YWNrRGVwdGggPSAzLApzdGF0dXMgPSAiYWN0aXZlIiwKdXNlck5hbWUg
    _ModuleTable008_=PSAiSERGNS8xLjEyLjAiLAp3ViA9ICIwMDAwMDAwMDEuMDAwMDAwMDEyLip6ZmluYWwiLAp9LApKYXZhID0gewpmbiA9ICIvc29mdHdhcmUva3A5MjAtUkw4L1N0YWdlcy8yMDIxYS9VSS9Ub29scy9KYXZhLzExLjAuMTAubHVhIiwKZnVsbE5hbWUgPSAiSmF2YS8xMS4wLjEwIiwKbG9hZE9yZGVyID0gMzMsCnByb3BUID0ge30sCnN0YWNrRGVwdGggPSAzLApzdGF0dXMgPSAiYWN0aXZlIiwKdXNlck5hbWUgPSAiSmF2YS8xMS4wLjEwIiwKd1YgPSAiMDAwMDAwMDExLjAwMDAwMDAwMC4wMDAwMDAwMTAuKnpmaW5hbCIsCn0sCk5BU00gPSB7CmZuID0gIi9zb2Z0d2FyZS9rcDkyMC1STDgvU3RhZ2VzLzIwMjFhL21vZHVsZXMvYWxsL0NvbXBpbGVyL0dDQ2NvcmUvOS4zLjAvTkFT
    _ModuleTable009_=TS8uMi4xNS4wNS5sdWEiLApmdWxsTmFtZSA9ICJOQVNNLy4yLjE1LjA1IiwKbG9hZE9yZGVyID0gMzksCnByb3BUID0ge30sCnN0YWNrRGVwdGggPSAzLApzdGF0dXMgPSAiYWN0aXZlIiwKdXNlck5hbWUgPSAiTkFTTS8uMi4xNS4wNSIsCndWID0gIjAwMDAwMDAwMC4wMDAwMDAwMDIuMDAwMDAwMDE1LjAwMDAwMDAwNS4qemZpbmFsIiwKfSwKT3BlbkJMQVMgPSB7CmZuID0gIi9zb2Z0d2FyZS9rcDkyMC1STDgvU3RhZ2VzLzIwMjFhL21vZHVsZXMvYWxsL0NvbXBpbGVyL0dDQy85LjMuMC9PcGVuQkxBUy8wLjMuMTkubHVhIiwKZnVsbE5hbWUgPSAiT3BlbkJMQVMvMC4zLjE5IiwKbG9hZE9yZGVyID0gMjMsCnByb3BUID0ge30sCnN0YWNrRGVwdGggPSAxLApzdGF0dXMgPSAi
    _ModuleTable010_=YWN0aXZlIiwKdXNlck5hbWUgPSAiT3BlbkJMQVMvMC4zLjE5IiwKd1YgPSAiMDAwMDAwMDAwLjAwMDAwMDAwMy4wMDAwMDAwMTkuKnpmaW5hbCIsCn0sCk9wZW5NUEkgPSB7CmZuID0gIi9zb2Z0d2FyZS9rcDkyMC1STDgvU3RhZ2VzLzIwMjFhL21vZHVsZXMvYWxsL0NvbXBpbGVyL21waS9HQ0MvOS4zLjAvT3Blbk1QSS80LjEuMi5sdWEiLApmdWxsTmFtZSA9ICJPcGVuTVBJLzQuMS4yIiwKbG9hZE9yZGVyID0gMjIsCnByb3BUID0ge30sCnN0YWNrRGVwdGggPSAwLApzdGF0dXMgPSAiYWN0aXZlIiwKdXNlck5hbWUgPSAiT3Blbk1QSSIsCndWID0gIjAwMDAwMDAwNC4wMDAwMDAwMDEuMDAwMDAwMDAyLip6ZmluYWwiLAp9LApQQVBJID0gewpmbiA9ICIvc29mdHdhcmUva3A5
    _ModuleTable011_=MjAtUkw4L1N0YWdlcy8yMDIxYS9tb2R1bGVzL2FsbC9Db21waWxlci9HQ0Njb3JlLzkuMy4wL1BBUEkvNi4wLjAuMS03MDg4N2RmNy5sdWEiLApmdWxsTmFtZSA9ICJQQVBJLzYuMC4wLjEtNzA4ODdkZjciLApsb2FkT3JkZXIgPSA2MCwKcHJvcFQgPSB7fSwKc3RhY2tEZXB0aCA9IDAsCnN0YXR1cyA9ICJhY3RpdmUiLAp1c2VyTmFtZSA9ICJQQVBJIiwKd1YgPSAiMDAwMDAwMDA2LjAwMDAwMDAwMC4wMDAwMDAwMDAuMDAwMDAwMDAxLip6ZmluYWwtLjAwMDA3MDg4Ny4qZGYuMDAwMDAwMDA3Lip6ZmluYWwiLAp9LApQb3N0Z3JlU1FMID0gewpmbiA9ICIvc29mdHdhcmUva3A5MjAtUkw4L1N0YWdlcy8yMDIxYS9tb2R1bGVzL2FsbC9Db21waWxlci9HQ0Njb3JlLzkuMy4wL1Bv
    _ModuleTable012_=c3RncmVTUUwvMTMuMi5sdWEiLApmdWxsTmFtZSA9ICJQb3N0Z3JlU1FMLzEzLjIiLApsb2FkT3JkZXIgPSAzNCwKcHJvcFQgPSB7fSwKc3RhY2tEZXB0aCA9IDIsCnN0YXR1cyA9ICJhY3RpdmUiLAp1c2VyTmFtZSA9ICJQb3N0Z3JlU1FMLzEzLjIiLAp3ViA9ICIwMDAwMDAwMTMuMDAwMDAwMDAyLip6ZmluYWwiLAp9LApQeXRob24gPSB7CmZuID0gIi9zb2Z0d2FyZS9rcDkyMC1STDgvU3RhZ2VzLzIwMjFhL21vZHVsZXMvYWxsL0NvbXBpbGVyL0dDQ2NvcmUvOS4zLjAvUHl0aG9uLzMuNy41Lmx1YSIsCmZ1bGxOYW1lID0gIlB5dGhvbi8zLjcuNSIsCmxvYWRPcmRlciA9IDQ1LApwcm9wVCA9IHt9LApzdGFja0RlcHRoID0gMSwKc3RhdHVzID0gImFjdGl2ZSIsCnVzZXJOYW1l
    _ModuleTable013_=ID0gIlB5dGhvbi8zLjcuNSIsCndWID0gIjAwMDAwMDAwMy4wMDAwMDAwMDcuMDAwMDAwMDA1Lip6ZmluYWwiLAp9LApSdXN0ID0gewpmbiA9ICIvc29mdHdhcmUva3A5MjAtUkw4L1N0YWdlcy8yMDIxYS9tb2R1bGVzL2FsbC9Db21waWxlci9HQ0Njb3JlLzkuMy4wL1J1c3QvMS41Mi4xLmx1YSIsCmZ1bGxOYW1lID0gIlJ1c3QvMS41Mi4xIiwKbG9hZE9yZGVyID0gMzUsCnByb3BUID0ge30sCnN0YWNrRGVwdGggPSAyLApzdGF0dXMgPSAiYWN0aXZlIiwKdXNlck5hbWUgPSAiUnVzdC8xLjUyLjEiLAp3ViA9ICIwMDAwMDAwMDEuMDAwMDAwMDUyLjAwMDAwMDAwMS4qemZpbmFsIiwKfSwKU1FMaXRlID0gewpmbiA9ICIvc29mdHdhcmUva3A5MjAtUkw4L1N0YWdlcy8yMDIxYS9t
    _ModuleTable014_=b2R1bGVzL2FsbC9Db21waWxlci9HQ0Njb3JlLzkuMy4wL1NRTGl0ZS8uMy4zNS4zLmx1YSIsCmZ1bGxOYW1lID0gIlNRTGl0ZS8uMy4zNS4zIiwKbG9hZE9yZGVyID0gMjgsCnByb3BUID0ge30sCnN0YWNrRGVwdGggPSAyLApzdGF0dXMgPSAiYWN0aXZlIiwKdXNlck5hbWUgPSAiU1FMaXRlLy4zLjM1LjMiLAp3ViA9ICIwMDAwMDAwMDAuMDAwMDAwMDAzLjAwMDAwMDAzNS4wMDAwMDAwMDMuKnpmaW5hbCIsCn0sClNjYUxBUEFDSyA9IHsKZm4gPSAiL3NvZnR3YXJlL2twOTIwLVJMOC9TdGFnZXMvMjAyMWEvbW9kdWxlcy9hbGwvTVBJL0dDQy85LjMuMC9PcGVuTVBJLzQuMS4yL1NjYUxBUEFDSy8yLjEuMC1PcGVuQkxBUy0wLjMuMTkubHVhIiwKZnVsbE5hbWUgPSAiU2NhTEFQ
    _ModuleTable015_=QUNLLzIuMS4wLU9wZW5CTEFTLTAuMy4xOSIsCmxvYWRPcmRlciA9IDI1LApwcm9wVCA9IHt9LApzdGFja0RlcHRoID0gMSwKc3RhdHVzID0gImFjdGl2ZSIsCnVzZXJOYW1lID0gIlNjYUxBUEFDSy8yLjEuMC1PcGVuQkxBUy0wLjMuMTkiLAp3ViA9ICIwMDAwMDAwMDIuMDAwMDAwMDAxLipvcGVuYmxhcy4qemZpbmFsLS4wMDAwMDAwMDAuMDAwMDAwMDAzLjAwMDAwMDAxOS4qemZpbmFsIiwKfSwKWyJTY2lQeS1TdGFjayJdID0gewpmbiA9ICIvc29mdHdhcmUva3A5MjAtUkw4L1N0YWdlcy8yMDIxYS9tb2R1bGVzL2FsbC9NUEkvR0NDLzkuMy4wL09wZW5NUEkvNC4xLjIvU2NpUHktU3RhY2svMjAyMWEtUHl0aG9uLTMuNy41Lmx1YSIsCmZ1bGxOYW1lID0gIlNjaVB5LVN0YWNr
    _ModuleTable016_=LzIwMjFhLVB5dGhvbi0zLjcuNSIsCmxvYWRPcmRlciA9IDQ3LApwcm9wVCA9IHt9LApzdGFja0RlcHRoID0gMSwKc3RhdHVzID0gImFjdGl2ZSIsCnVzZXJOYW1lID0gIlNjaVB5LVN0YWNrLzIwMjFhLVB5dGhvbi0zLjcuNSIsCndWID0gIjAwMDAwMjAyMS4qYS4qeXRob24uKnpmaW5hbC0uMDAwMDAwMDAzLjAwMDAwMDAwNy4wMDAwMDAwMDUuKnpmaW5hbCIsCn0sClN6aXAgPSB7CmZuID0gIi9zb2Z0d2FyZS9rcDkyMC1STDgvU3RhZ2VzLzIwMjFhL21vZHVsZXMvYWxsL0NvbXBpbGVyL0dDQ2NvcmUvOS4zLjAvU3ppcC8uMi4xLjEubHVhIiwKZnVsbE5hbWUgPSAiU3ppcC8uMi4xLjEiLApsb2FkT3JkZXIgPSA0OSwKcHJvcFQgPSB7fSwKc3RhY2tEZXB0aCA9IDQsCnN0YXR1
    _ModuleTable017_=cyA9ICJhY3RpdmUiLAp1c2VyTmFtZSA9ICJTemlwLy4yLjEuMSIsCndWID0gIjAwMDAwMDAwMC4wMDAwMDAwMDIuMDAwMDAwMDAxLjAwMDAwMDAwMS4qemZpbmFsIiwKfSwKVGNsID0gewpmbiA9ICIvc29mdHdhcmUva3A5MjAtUkw4L1N0YWdlcy8yMDIxYS9tb2R1bGVzL2FsbC9Db21waWxlci9HQ0Njb3JlLzkuMy4wL1RjbC84LjYuMTEubHVhIiwKZnVsbE5hbWUgPSAiVGNsLzguNi4xMSIsCmxvYWRPcmRlciA9IDI3LApwcm9wVCA9IHt9LApzdGFja0RlcHRoID0gMywKc3RhdHVzID0gImFjdGl2ZSIsCnVzZXJOYW1lID0gIlRjbC84LjYuMTEiLAp3ViA9ICIwMDAwMDAwMDguMDAwMDAwMDA2LjAwMDAwMDAxMS4qemZpbmFsIiwKfSwKWyJUZW5zb3JGbG93LUNBTk4iXSA9IHsK
    _ModuleTable018_=Zm4gPSAiL3NvZnR3YXJlL2twOTIwLVJMOC9TdGFnZXMvMjAyMWEvbW9kdWxlcy9hbGwvTVBJL0dDQy85LjMuMC9PcGVuTVBJLzQuMS4yL1RlbnNvckZsb3ctQ0FOTi8xLjE1LjAtUHl0aG9uLTMuNy41Lmx1YSIsCmZ1bGxOYW1lID0gIlRlbnNvckZsb3ctQ0FOTi8xLjE1LjAtUHl0aG9uLTMuNy41IiwKbG9hZE9yZGVyID0gNTcsCnByb3BUID0ge30sCnN0YWNrRGVwdGggPSAwLApzdGF0dXMgPSAiYWN0aXZlIiwKdXNlck5hbWUgPSAiVGVuc29yRmxvdy1DQU5OIiwKd1YgPSAiMDAwMDAwMDAxLjAwMDAwMDAxNS4qeXRob24uKnpmaW5hbC0uMDAwMDAwMDAzLjAwMDAwMDAwNy4wMDAwMDAwMDUuKnpmaW5hbCIsCn0sClsiVGVuc29yRmxvdy1DQU5OLUNvcmUiXSA9IHsKZm4gPSAi
    _ModuleTable019_=L3NvZnR3YXJlL2twOTIwLVJMOC9TdGFnZXMvMjAyMWEvbW9kdWxlcy9hbGwvTVBJL0dDQy85LjMuMC9PcGVuTVBJLzQuMS4yL1RlbnNvckZsb3ctQ0FOTi1Db3JlLzEuMTUuMC1QeXRob24tMy43LjUubHVhIiwKZnVsbE5hbWUgPSAiVGVuc29yRmxvdy1DQU5OLUNvcmUvMS4xNS4wLVB5dGhvbi0zLjcuNSIsCmxvYWRPcmRlciA9IDUzLApwcm9wVCA9IHt9LApzdGFja0RlcHRoID0gMSwKc3RhdHVzID0gImFjdGl2ZSIsCnVzZXJOYW1lID0gIlRlbnNvckZsb3ctQ0FOTi1Db3JlLzEuMTUuMC1QeXRob24tMy43LjUiLAp3ViA9ICIwMDAwMDAwMDEuMDAwMDAwMDE1Lip5dGhvbi4qemZpbmFsLS4wMDAwMDAwMDMuMDAwMDAwMDA3LjAwMDAwMDAwNS4qemZpbmFsIiwKfSwKVUNYID0g
    _ModuleTable020_=ewpmbiA9ICIvc29mdHdhcmUva3A5MjAtUkw4L1N0YWdlcy8yMDIxYS9tb2R1bGVzL2FsbC9Db21waWxlci9HQ0Njb3JlLzkuMy4wL1VDWC8xLjExLjIubHVhIiwKZnVsbE5hbWUgPSAiVUNYLzEuMTEuMiIsCmxvYWRPcmRlciA9IDIxLApwcm9wVCA9IHt9LApzdGFja0RlcHRoID0gMSwKc3RhdHVzID0gImFjdGl2ZSIsCnVzZXJOYW1lID0gIlVDWC8xLjExLjIiLAp3ViA9ICIwMDAwMDAwMDEuMDAwMDAwMDExLjAwMDAwMDAwMi4qemZpbmFsIiwKfSwKWDExID0gewpmbiA9ICIvc29mdHdhcmUva3A5MjAtUkw4L1N0YWdlcy8yMDIxYS9tb2R1bGVzL2FsbC9Db21waWxlci9HQ0Njb3JlLzkuMy4wL1gxMS8yMDIxMDMzMS5sdWEiLApmdWxsTmFtZSA9ICJYMTEvMjAyMTAzMzEiLAps
    _ModuleTable021_=b2FkT3JkZXIgPSAxOSwKcHJvcFQgPSB7fSwKc3RhY2tEZXB0aCA9IDIsCnN0YXR1cyA9ICJhY3RpdmUiLAp1c2VyTmFtZSA9ICJYMTEvMjAyMTAzMzEiLAp3ViA9ICIwMjAyMTAzMzEuKnpmaW5hbCIsCn0sClhaID0gewpmbiA9ICIvc29mdHdhcmUva3A5MjAtUkw4L1N0YWdlcy8yMDIxYS9tb2R1bGVzL2FsbC9Db21waWxlci9HQ0Njb3JlLzkuMy4wL1haLy41LjIuNS5sdWEiLApmdWxsTmFtZSA9ICJYWi8uNS4yLjUiLApsb2FkT3JkZXIgPSAxMCwKcHJvcFQgPSB7fSwKc3RhY2tEZXB0aCA9IDMsCnN0YXR1cyA9ICJhY3RpdmUiLAp1c2VyTmFtZSA9ICJYWi8uNS4yLjUiLAp3ViA9ICIwMDAwMDAwMDAuMDAwMDAwMDA1LjAwMDAwMDAwMi4wMDAwMDAwMDUuKnpmaW5hbCIsCn0s
    _ModuleTable022_=ClsiYWJzZWlsLWNwcCJdID0gewpmbiA9ICIvc29mdHdhcmUva3A5MjAtUkw4L1N0YWdlcy8yMDIxYS9tb2R1bGVzL2FsbC9Db21waWxlci9HQ0Njb3JlLzkuMy4wL2Fic2VpbC1jcHAvMjAyMTAzMjQuMS5sdWEiLApmdWxsTmFtZSA9ICJhYnNlaWwtY3BwLzIwMjEwMzI0LjEiLApsb2FkT3JkZXIgPSA0MSwKcHJvcFQgPSB7fSwKc3RhY2tEZXB0aCA9IDMsCnN0YXR1cyA9ICJhY3RpdmUiLAp1c2VyTmFtZSA9ICJhYnNlaWwtY3BwLzIwMjEwMzI0LjEiLAp3ViA9ICIwMjAyMTAzMjQuMDAwMDAwMDAxLip6ZmluYWwiLAp9LApbImFybS1vcHRpbWl6ZWQtcm91dGluZXMiXSA9IHsKZm4gPSAiL3NvZnR3YXJlL21vZHVsZXMvY3VzdG9tL2FybS1vcHRpbWl6ZWQtcm91dGluZXMvMjEu
    _ModuleTable023_=MDIubHVhIiwKZnVsbE5hbWUgPSAiYXJtLW9wdGltaXplZC1yb3V0aW5lcy8yMS4wMiIsCmxvYWRPcmRlciA9IDMsCnByb3BUID0ge30sCnN0YWNrRGVwdGggPSAxLApzdGF0dXMgPSAiYWN0aXZlIiwKdXNlck5hbWUgPSAiYXJtLW9wdGltaXplZC1yb3V0aW5lcyIsCndWID0gIjAwMDAwMDAyMS4wMDAwMDAwMDIuKnpmaW5hbCIsCn0sCmJpbnV0aWxzID0gewpmbiA9ICIvc29mdHdhcmUva3A5MjAtUkw4L1N0YWdlcy8yMDIxYS9tb2R1bGVzL2FsbC9Db21waWxlci9HQ0Njb3JlLzkuMy4wL2JpbnV0aWxzLy4yLjM2LjEubHVhIiwKZnVsbE5hbWUgPSAiYmludXRpbHMvLjIuMzYuMSIsCmxvYWRPcmRlciA9IDYsCnByb3BUID0ge30sCnN0YWNrRGVwdGggPSAxLApzdGF0dXMgPSAi
    _ModuleTable024_=YWN0aXZlIiwKdXNlck5hbWUgPSAiYmludXRpbHMvLjIuMzYuMSIsCndWID0gIjAwMDAwMDAwMC4wMDAwMDAwMDIuMDAwMDAwMDM2LjAwMDAwMDAwMS4qemZpbmFsIiwKfSwKYnppcDIgPSB7CmZuID0gIi9zb2Z0d2FyZS9rcDkyMC1STDgvU3RhZ2VzLzIwMjFhL21vZHVsZXMvYWxsL0NvbXBpbGVyL0dDQ2NvcmUvOS4zLjAvYnppcDIvLjEuMC44Lmx1YSIsCmZ1bGxOYW1lID0gImJ6aXAyLy4xLjAuOCIsCmxvYWRPcmRlciA9IDEyLApwcm9wVCA9IHt9LApzdGFja0RlcHRoID0gMywKc3RhdHVzID0gImFjdGl2ZSIsCnVzZXJOYW1lID0gImJ6aXAyLy4xLjAuOCIsCndWID0gIjAwMDAwMDAwMC4wMDAwMDAwMDEuMDAwMDAwMDAwLjAwMDAwMDAwOC4qemZpbmFsIiwKfSwKWyJjLWFy
    _ModuleTable025_=ZXMiXSA9IHsKZm4gPSAiL3NvZnR3YXJlL2twOTIwLVJMOC9TdGFnZXMvMjAyMWEvbW9kdWxlcy9hbGwvQ29tcGlsZXIvR0NDY29yZS85LjMuMC9jLWFyZXMvMS4xNy4xLmx1YSIsCmZ1bGxOYW1lID0gImMtYXJlcy8xLjE3LjEiLApsb2FkT3JkZXIgPSA0MiwKcHJvcFQgPSB7fSwKc3RhY2tEZXB0aCA9IDMsCnN0YXR1cyA9ICJhY3RpdmUiLAp1c2VyTmFtZSA9ICJjLWFyZXMvMS4xNy4xIiwKd1YgPSAiMDAwMDAwMDAxLjAwMDAwMDAxNy4wMDAwMDAwMDEuKnpmaW5hbCIsCn0sCmV4cGF0ID0gewpmbiA9ICIvc29mdHdhcmUva3A5MjAtUkw4L1N0YWdlcy8yMDIxYS9tb2R1bGVzL2FsbC9Db21waWxlci9HQ0Njb3JlLzkuMy4wL2V4cGF0Ly4yLjQuMi5sdWEiLApmdWxsTmFtZSA9
    _ModuleTable026_=ICJleHBhdC8uMi40LjIiLApsb2FkT3JkZXIgPSAxMywKcHJvcFQgPSB7fSwKc3RhY2tEZXB0aCA9IDQsCnN0YXR1cyA9ICJhY3RpdmUiLAp1c2VyTmFtZSA9ICJleHBhdC8uMi40LjIiLAp3ViA9ICIwMDAwMDAwMDAuMDAwMDAwMDAyLjAwMDAwMDAwNC4wMDAwMDAwMDIuKnpmaW5hbCIsCn0sCmZvbnRjb25maWcgPSB7CmZuID0gIi9zb2Z0d2FyZS9rcDkyMC1STDgvU3RhZ2VzLzIwMjFhL21vZHVsZXMvYWxsL0NvbXBpbGVyL0dDQ2NvcmUvOS4zLjAvZm9udGNvbmZpZy8uMi4xMy45My5sdWEiLApmdWxsTmFtZSA9ICJmb250Y29uZmlnLy4yLjEzLjkzIiwKbG9hZE9yZGVyID0gMTgsCnByb3BUID0ge30sCnN0YWNrRGVwdGggPSAzLApzdGF0dXMgPSAiYWN0aXZlIiwKdXNlck5h
    _ModuleTable027_=bWUgPSAiZm9udGNvbmZpZy8uMi4xMy45MyIsCndWID0gIjAwMDAwMDAwMC4wMDAwMDAwMDIuMDAwMDAwMDEzLjAwMDAwMDA5My4qemZpbmFsIiwKfSwKZnJlZXR5cGUgPSB7CmZuID0gIi9zb2Z0d2FyZS9rcDkyMC1STDgvU3RhZ2VzLzIwMjFhL21vZHVsZXMvYWxsL0NvbXBpbGVyL0dDQ2NvcmUvOS4zLjAvZnJlZXR5cGUvLjIuMTAuNC5sdWEiLApmdWxsTmFtZSA9ICJmcmVldHlwZS8uMi4xMC40IiwKbG9hZE9yZGVyID0gMTUsCnByb3BUID0ge30sCnN0YWNrRGVwdGggPSA0LApzdGF0dXMgPSAiYWN0aXZlIiwKdXNlck5hbWUgPSAiZnJlZXR5cGUvLjIuMTAuNCIsCndWID0gIjAwMDAwMDAwMC4wMDAwMDAwMDIuMDAwMDAwMDEwLjAwMDAwMDAwNC4qemZpbmFsIiwKfSwKZ1JQ
    _ModuleTable028_=QyA9IHsKZm4gPSAiL3NvZnR3YXJlL2twOTIwLVJMOC9TdGFnZXMvMjAyMWEvbW9kdWxlcy9hbGwvQ29tcGlsZXIvR0NDY29yZS85LjMuMC9nUlBDLzEuMzcuMS5sdWEiLApmdWxsTmFtZSA9ICJnUlBDLzEuMzcuMSIsCmxvYWRPcmRlciA9IDQ0LApwcm9wVCA9IHt9LApzdGFja0RlcHRoID0gMiwKc3RhdHVzID0gImFjdGl2ZSIsCnVzZXJOYW1lID0gImdSUEMvMS4zNy4xIiwKd1YgPSAiMDAwMDAwMDAxLjAwMDAwMDAzNy4wMDAwMDAwMDEuKnpmaW5hbCIsCn0sCmdmbGFncyA9IHsKZm4gPSAiL3NvZnR3YXJlL2twOTIwLVJMOC9TdGFnZXMvMjAyMWEvbW9kdWxlcy9hbGwvQ29tcGlsZXIvR0NDY29yZS85LjMuMC9nZmxhZ3MvLjIuMi4yLmx1YSIsCmZ1bGxOYW1lID0gImdmbGFn
    _ModuleTable029_=cy8uMi4yLjIiLApsb2FkT3JkZXIgPSAzNywKcHJvcFQgPSB7fSwKc3RhY2tEZXB0aCA9IDIsCnN0YXR1cyA9ICJhY3RpdmUiLAp1c2VyTmFtZSA9ICJnZmxhZ3MvLjIuMi4yIiwKd1YgPSAiMDAwMDAwMDAwLjAwMDAwMDAwMi4wMDAwMDAwMDIuMDAwMDAwMDAyLip6ZmluYWwiLAp9LApncGVyZiA9IHsKZm4gPSAiL3NvZnR3YXJlL2twOTIwLVJMOC9TdGFnZXMvMjAyMWEvbW9kdWxlcy9hbGwvQ29tcGlsZXIvR0NDY29yZS85LjMuMC9ncGVyZi8uMy4xLmx1YSIsCmZ1bGxOYW1lID0gImdwZXJmLy4zLjEiLApsb2FkT3JkZXIgPSAxNiwKcHJvcFQgPSB7fSwKc3RhY2tEZXB0aCA9IDQsCnN0YXR1cyA9ICJhY3RpdmUiLAp1c2VyTmFtZSA9ICJncGVyZi8uMy4xIiwKd1YgPSAiMDAw
    _ModuleTable030_=MDAwMDAwLjAwMDAwMDAwMy4wMDAwMDAwMDEuKnpmaW5hbCIsCn0sCmg1cHkgPSB7CmZuID0gIi9zb2Z0d2FyZS9rcDkyMC1STDgvU3RhZ2VzLzIwMjFhL21vZHVsZXMvYWxsL01QSS9HQ0MvOS4zLjAvT3Blbk1QSS80LjEuMi9oNXB5LzMuNS4wLVB5dGhvbi0zLjcuNS5sdWEiLApmdWxsTmFtZSA9ICJoNXB5LzMuNS4wLVB5dGhvbi0zLjcuNSIsCmxvYWRPcmRlciA9IDUxLApwcm9wVCA9IHt9LApzdGFja0RlcHRoID0gMiwKc3RhdHVzID0gImFjdGl2ZSIsCnVzZXJOYW1lID0gImg1cHkvMy41LjAtUHl0aG9uLTMuNy41IiwKd1YgPSAiMDAwMDAwMDAzLjAwMDAwMDAwNS4qeXRob24uKnpmaW5hbC0uMDAwMDAwMDAzLjAwMDAwMDAwNy4wMDAwMDAwMDUuKnpmaW5hbCIsCn0sCmh3
    _ModuleTable031_=bG9jID0gewpmbiA9ICIvc29mdHdhcmUva3A5MjAtUkw4L1N0YWdlcy8yMDIxYS9tb2R1bGVzL2FsbC9Db21waWxlci9HQ0Njb3JlLzkuMy4wL2h3bG9jLzIuNC4xLmx1YSIsCmZ1bGxOYW1lID0gImh3bG9jLzIuNC4xIiwKbG9hZE9yZGVyID0gMjAsCnByb3BUID0ge30sCnN0YWNrRGVwdGggPSAxLApzdGF0dXMgPSAiYWN0aXZlIiwKdXNlck5hbWUgPSAiaHdsb2MvMi40LjEiLAp3ViA9ICIwMDAwMDAwMDIuMDAwMDAwMDA0LjAwMDAwMDAwMS4qemZpbmFsIiwKfSwKbGliZmZpID0gewpmbiA9ICIvc29mdHdhcmUva3A5MjAtUkw4L1N0YWdlcy8yMDIxYS9tb2R1bGVzL2FsbC9Db21waWxlci9HQ0Njb3JlLzkuMy4wL2xpYmZmaS8zLjMubHVhIiwKZnVsbE5hbWUgPSAibGliZmZp
    _ModuleTable032_=LzMuMyIsCmxvYWRPcmRlciA9IDMxLApwcm9wVCA9IHt9LApzdGFja0RlcHRoID0gMiwKc3RhdHVzID0gImFjdGl2ZSIsCnVzZXJOYW1lID0gImxpYmZmaS8zLjMiLAp3ViA9ICIwMDAwMDAwMDMuMDAwMDAwMDAzLip6ZmluYWwiLAp9LApbImxpYmpwZWctdHVyYm8iXSA9IHsKZm4gPSAiL3NvZnR3YXJlL2twOTIwLVJMOC9TdGFnZXMvMjAyMWEvbW9kdWxlcy9hbGwvQ29tcGlsZXIvR0NDY29yZS85LjMuMC9saWJqcGVnLXR1cmJvLy4yLjAuNi5sdWEiLApmdWxsTmFtZSA9ICJsaWJqcGVnLXR1cmJvLy4yLjAuNiIsCmxvYWRPcmRlciA9IDQwLApwcm9wVCA9IHt9LApzdGFja0RlcHRoID0gMiwKc3RhdHVzID0gImFjdGl2ZSIsCnVzZXJOYW1lID0gImxpYmpwZWctdHVyYm8vLjIu
    _ModuleTable033_=MC42IiwKd1YgPSAiMDAwMDAwMDAwLjAwMDAwMDAwMi4wMDAwMDAwMDAuMDAwMDAwMDA2Lip6ZmluYWwiLAp9LApsaWJwZm0gPSB7CmZuID0gIi9zb2Z0d2FyZS9rcDkyMC1STDgvU3RhZ2VzLzIwMjFhL21vZHVsZXMvYWxsL0NvbXBpbGVyL0dDQ2NvcmUvOS4zLjAvbGlicGZtLzQuMTEuMS1mNjUwMGU3Ny5sdWEiLApmdWxsTmFtZSA9ICJsaWJwZm0vNC4xMS4xLWY2NTAwZTc3IiwKbG9hZE9yZGVyID0gNTksCnByb3BUID0ge30sCnN0YWNrRGVwdGggPSAxLApzdGF0dXMgPSAiYWN0aXZlIiwKdXNlck5hbWUgPSAibGlicGZtLzQuMTEuMS1mNjUwMGU3NyIsCndWID0gIjAwMDAwMDAwNC4wMDAwMDAwMTEuMDAwMDAwMDAxLipmLjAwMDAwNjUwMC4qZS4wMDAwMDAwNzcuKnpmaW5h
    _ModuleTable034_=bCIsCn0sCmxpYnBuZyA9IHsKZm4gPSAiL3NvZnR3YXJlL2twOTIwLVJMOC9TdGFnZXMvMjAyMWEvbW9kdWxlcy9hbGwvQ29tcGlsZXIvR0NDY29yZS85LjMuMC9saWJwbmcvLjEuNi4zNy5sdWEiLApmdWxsTmFtZSA9ICJsaWJwbmcvLjEuNi4zNyIsCmxvYWRPcmRlciA9IDE0LApwcm9wVCA9IHt9LApzdGFja0RlcHRoID0gNSwKc3RhdHVzID0gImFjdGl2ZSIsCnVzZXJOYW1lID0gImxpYnBuZy8uMS42LjM3IiwKd1YgPSAiMDAwMDAwMDAwLjAwMDAwMDAwMS4wMDAwMDAwMDYuMDAwMDAwMDM3Lip6ZmluYWwiLAp9LApsaWJyZWFkbGluZSA9IHsKZm4gPSAiL3NvZnR3YXJlL2twOTIwLVJMOC9TdGFnZXMvMjAyMWEvbW9kdWxlcy9hbGwvQ29tcGlsZXIvR0NDY29yZS85LjMuMC9s
    _ModuleTable035_=aWJyZWFkbGluZS8uOC4xLmx1YSIsCmZ1bGxOYW1lID0gImxpYnJlYWRsaW5lLy44LjEiLApsb2FkT3JkZXIgPSAyNiwKcHJvcFQgPSB7fSwKc3RhY2tEZXB0aCA9IDIsCnN0YXR1cyA9ICJhY3RpdmUiLAp1c2VyTmFtZSA9ICJsaWJyZWFkbGluZS8uOC4xIiwKd1YgPSAiMDAwMDAwMDAwLjAwMDAwMDAwOC4wMDAwMDAwMDEuKnpmaW5hbCIsCn0sCmxpYnNwYXRpYWxpbmRleCA9IHsKZm4gPSAiL3NvZnR3YXJlL2twOTIwLVJMOC9TdGFnZXMvMjAyMWEvbW9kdWxlcy9hbGwvQ29tcGlsZXIvR0NDY29yZS85LjMuMC9saWJzcGF0aWFsaW5kZXgvLjEuOS4zLmx1YSIsCmZ1bGxOYW1lID0gImxpYnNwYXRpYWxpbmRleC8uMS45LjMiLApsb2FkT3JkZXIgPSAzOCwKcHJvcFQgPSB7fSwK
    _ModuleTable036_=c3RhY2tEZXB0aCA9IDIsCnN0YXR1cyA9ICJhY3RpdmUiLAp1c2VyTmFtZSA9ICJsaWJzcGF0aWFsaW5kZXgvLjEuOS4zIiwKd1YgPSAiMDAwMDAwMDAwLjAwMDAwMDAwMS4wMDAwMDAwMDkuMDAwMDAwMDAzLip6ZmluYWwiLAp9LApsaWJ4bWwyID0gewpmbiA9ICIvc29mdHdhcmUva3A5MjAtUkw4L1N0YWdlcy8yMDIxYS9tb2R1bGVzL2FsbC9Db21waWxlci9HQ0Njb3JlLzkuMy4wL2xpYnhtbDIvLjIuOS4xMC1iYWJlNzUwMy5sdWEiLApmdWxsTmFtZSA9ICJsaWJ4bWwyLy4yLjkuMTAtYmFiZTc1MDMiLApsb2FkT3JkZXIgPSAxMSwKcHJvcFQgPSB7fSwKc3RhY2tEZXB0aCA9IDIsCnN0YXR1cyA9ICJhY3RpdmUiLAp1c2VyTmFtZSA9ICJsaWJ4bWwyLy4yLjkuMTAtYmFiZTc1
    _ModuleTable037_=MDMiLAp3ViA9ICIwMDAwMDAwMDAuMDAwMDAwMDAyLjAwMDAwMDAwOS4wMDAwMDAwMTAuKmJhYmUuMDAwMDA3NTAzLip6ZmluYWwiLAp9LApsaWJ4c2x0ID0gewpmbiA9ICIvc29mdHdhcmUva3A5MjAtUkw4L1N0YWdlcy8yMDIxYS9tb2R1bGVzL2FsbC9Db21waWxlci9HQ0Njb3JlLzkuMy4wL2xpYnhzbHQvLjEuMS4zNC5sdWEiLApmdWxsTmFtZSA9ICJsaWJ4c2x0Ly4xLjEuMzQiLApsb2FkT3JkZXIgPSAzMCwKcHJvcFQgPSB7fSwKc3RhY2tEZXB0aCA9IDIsCnN0YXR1cyA9ICJhY3RpdmUiLAp1c2VyTmFtZSA9ICJsaWJ4c2x0Ly4xLjEuMzQiLAp3ViA9ICIwMDAwMDAwMDAuMDAwMDAwMDAxLjAwMDAwMDAwMS4wMDAwMDAwMzQuKnpmaW5hbCIsCn0sCmxpYnlhbWwgPSB7CmZu
    _ModuleTable038_=ID0gIi9zb2Z0d2FyZS9rcDkyMC1STDgvU3RhZ2VzLzIwMjFhL21vZHVsZXMvYWxsL0NvbXBpbGVyL0dDQ2NvcmUvOS4zLjAvbGlieWFtbC8wLjIuNS5sdWEiLApmdWxsTmFtZSA9ICJsaWJ5YW1sLzAuMi41IiwKbG9hZE9yZGVyID0gMzIsCnByb3BUID0ge30sCnN0YWNrRGVwdGggPSAyLApzdGF0dXMgPSAiYWN0aXZlIiwKdXNlck5hbWUgPSAibGlieWFtbC8wLjIuNSIsCndWID0gIjAwMDAwMDAwMC4wMDAwMDAwMDIuMDAwMDAwMDA1Lip6ZmluYWwiLAp9LApuY3Vyc2VzID0gewpmbiA9ICIvc29mdHdhcmUva3A5MjAtUkw4L1N0YWdlcy8yMDIxYS9tb2R1bGVzL2FsbC9Db21waWxlci9HQ0Njb3JlLzkuMy4wL25jdXJzZXMvLjYuMi5sdWEiLApmdWxsTmFtZSA9ICJuY3Vyc2Vz
    _ModuleTable039_=Ly42LjIiLApsb2FkT3JkZXIgPSA1OCwKcHJvcFQgPSB7fSwKc3RhY2tEZXB0aCA9IDAsCnN0YXR1cyA9ICJhY3RpdmUiLAp1c2VyTmFtZSA9ICJuY3Vyc2VzLy42LjIiLAp3ViA9ICIwMDAwMDAwMDAuMDAwMDAwMDA2LjAwMDAwMDAwMi4qemZpbmFsIiwKfSwKWyJuc3luYy1DQU5OIl0gPSB7CmZuID0gIi9zb2Z0d2FyZS9rcDkyMC1STDgvU3RhZ2VzLzIwMjFhL21vZHVsZXMvYWxsL0NvbXBpbGVyL0dDQ2NvcmUvOS4zLjAvbnN5bmMtQ0FOTi8xLjIyLjAubHVhIiwKZnVsbE5hbWUgPSAibnN5bmMtQ0FOTi8xLjIyLjAiLApsb2FkT3JkZXIgPSA1MiwKcHJvcFQgPSB7fSwKc3RhY2tEZXB0aCA9IDIsCnN0YXR1cyA9ICJhY3RpdmUiLAp1c2VyTmFtZSA9ICJuc3luYy1DQU5OLzEu
    _ModuleTable040_=MjIuMCIsCndWID0gIjAwMDAwMDAwMS4wMDAwMDAwMjIuKnpmaW5hbCIsCn0sCm51bWFjdGwgPSB7CmZuID0gIi9zb2Z0d2FyZS9rcDkyMC1STDgvU3RhZ2VzLzIwMjFhL21vZHVsZXMvYWxsL0NvbXBpbGVyL0dDQ2NvcmUvOS4zLjAvbnVtYWN0bC8yLjAuMTQubHVhIiwKZnVsbE5hbWUgPSAibnVtYWN0bC8yLjAuMTQiLApsb2FkT3JkZXIgPSA5LApwcm9wVCA9IHt9LApzdGFja0RlcHRoID0gMiwKc3RhdHVzID0gImFjdGl2ZSIsCnVzZXJOYW1lID0gIm51bWFjdGwvMi4wLjE0IiwKd1YgPSAiMDAwMDAwMDAyLjAwMDAwMDAwMC4wMDAwMDAwMTQuKnpmaW5hbCIsCn0sCnByb3RvYnVmID0gewpmbiA9ICIvc29mdHdhcmUva3A5MjAtUkw4L1N0YWdlcy8yMDIxYS9tb2R1bGVzL2Fs
    _ModuleTable041_=bC9Db21waWxlci9HQ0Njb3JlLzkuMy4wL3Byb3RvYnVmLy4zLjE1LjYubHVhIiwKZnVsbE5hbWUgPSAicHJvdG9idWYvLjMuMTUuNiIsCmxvYWRPcmRlciA9IDM2LApwcm9wVCA9IHt9LApzdGFja0RlcHRoID0gMiwKc3RhdHVzID0gImFjdGl2ZSIsCnVzZXJOYW1lID0gInByb3RvYnVmLy4zLjE1LjYiLAp3ViA9ICIwMDAwMDAwMDAuMDAwMDAwMDAzLjAwMDAwMDAxNS4wMDAwMDAwMDYuKnpmaW5hbCIsCn0sClsicHJvdG9idWYtQ0FOTiJdID0gewpmbiA9ICIvc29mdHdhcmUva3A5MjAtUkw4L1N0YWdlcy8yMDIxYS9tb2R1bGVzL2FsbC9Db21waWxlci9HQ0Njb3JlLzkuMy4wL3Byb3RvYnVmLUNBTk4vMy4xNS42Lmx1YSIsCmZ1bGxOYW1lID0gInByb3RvYnVmLUNBTk4vMy4x
    _ModuleTable042_=NS42IiwKbG9hZE9yZGVyID0gNDgsCnByb3BUID0ge30sCnN0YWNrRGVwdGggPSAxLApzdGF0dXMgPSAiYWN0aXZlIiwKdXNlck5hbWUgPSAicHJvdG9idWYtQ0FOTi8zLjE1LjYiLAp3ViA9ICIwMDAwMDAwMDMuMDAwMDAwMDE1LjAwMDAwMDAwNi4qemZpbmFsIiwKfSwKcHliaW5kMTEgPSB7CmZuID0gIi9zb2Z0d2FyZS9rcDkyMC1STDgvU3RhZ2VzLzIwMjFhL21vZHVsZXMvYWxsL0NvbXBpbGVyL0dDQ2NvcmUvOS4zLjAvcHliaW5kMTEvLjIuNi4yLVB5dGhvbi0zLjcuNS5sdWEiLApmdWxsTmFtZSA9ICJweWJpbmQxMS8uMi42LjItUHl0aG9uLTMuNy41IiwKbG9hZE9yZGVyID0gNDYsCnByb3BUID0ge30sCnN0YWNrRGVwdGggPSAyLApzdGF0dXMgPSAiYWN0aXZlIiwKdXNl
    _ModuleTable043_=ck5hbWUgPSAicHliaW5kMTEvLjIuNi4yLVB5dGhvbi0zLjcuNSIsCndWID0gIjAwMDAwMDAwMC4wMDAwMDAwMDIuMDAwMDAwMDA2LjAwMDAwMDAwMi4qeXRob24uKnpmaW5hbC0uMDAwMDAwMDAzLjAwMDAwMDAwNy4wMDAwMDAwMDUuKnpmaW5hbCIsCn0sCnJlMiA9IHsKZm4gPSAiL3NvZnR3YXJlL2twOTIwLVJMOC9TdGFnZXMvMjAyMWEvbW9kdWxlcy9hbGwvQ29tcGlsZXIvR0NDY29yZS85LjMuMC9yZTIvMjAyMS0wNC0wMS5sdWEiLApmdWxsTmFtZSA9ICJyZTIvMjAyMS0wNC0wMSIsCmxvYWRPcmRlciA9IDQzLApwcm9wVCA9IHt9LApzdGFja0RlcHRoID0gMywKc3RhdHVzID0gImFjdGl2ZSIsCnVzZXJOYW1lID0gInJlMi8yMDIxLTA0LTAxIiwKd1YgPSAiMDAwMDAyMDIx
    _ModuleTable044_=Lip6ZmluYWwtLjAwMDAwMDAwNC4qemZpbmFsLS4wMDAwMDAwMDEuKnpmaW5hbCIsCn0sCnRtdXggPSB7CmZuID0gIi9zb2Z0d2FyZS9rcDkyMC1STDgvU3RhZ2VzLzIwMjFhL1VJL1Rvb2xzL3RtdXgvMy4yYS5sdWEiLApmdWxsTmFtZSA9ICJ0bXV4LzMuMmEiLApsb2FkT3JkZXIgPSAxLApwcm9wVCA9IHt9LApzdGFja0RlcHRoID0gMCwKc3RhdHVzID0gImFjdGl2ZSIsCnVzZXJOYW1lID0gInRtdXgiLAp3ViA9ICIwMDAwMDAwMDMuMDAwMDAwMDAyLiphLip6ZmluYWwiLAp9LApbInV0aWwtbGludXgiXSA9IHsKZm4gPSAiL3NvZnR3YXJlL2twOTIwLVJMOC9TdGFnZXMvMjAyMWEvbW9kdWxlcy9hbGwvQ29tcGlsZXIvR0NDY29yZS85LjMuMC91dGlsLWxpbnV4Ly4yLjM1LjEu
    _ModuleTable045_=bHVhIiwKZnVsbE5hbWUgPSAidXRpbC1saW51eC8uMi4zNS4xIiwKbG9hZE9yZGVyID0gMTcsCnByb3BUID0ge30sCnN0YWNrRGVwdGggPSA0LApzdGF0dXMgPSAiYWN0aXZlIiwKdXNlck5hbWUgPSAidXRpbC1saW51eC8uMi4zNS4xIiwKd1YgPSAiMDAwMDAwMDAwLjAwMDAwMDAwMi4wMDAwMDAwMzUuMDAwMDAwMDAxLip6ZmluYWwiLAp9LAp6bGliID0gewpmbiA9ICIvc29mdHdhcmUva3A5MjAtUkw4L1N0YWdlcy8yMDIxYS9tb2R1bGVzL2FsbC9Db21waWxlci9HQ0Njb3JlLzkuMy4wL3psaWIvLjEuMi4xMS5sdWEiLApmdWxsTmFtZSA9ICJ6bGliLy4xLjIuMTEiLApsb2FkT3JkZXIgPSA4LApwcm9wVCA9IHt9LApzdGFja0RlcHRoID0gMSwKc3RhdHVzID0gImFjdGl2ZSIs
    _ModuleTable046_=CnVzZXJOYW1lID0gInpsaWIvLjEuMi4xMSIsCndWID0gIjAwMDAwMDAwMC4wMDAwMDAwMDEuMDAwMDAwMDAyLjAwMDAwMDAxMS4qemZpbmFsIiwKfSwKenNoID0gewpmbiA9ICIvc29mdHdhcmUva3A5MjAtUkw4L1N0YWdlcy8yMDIxYS9VSS9Ub29scy96c2gvNS44Lmx1YSIsCmZ1bGxOYW1lID0gInpzaC81LjgiLApsb2FkT3JkZXIgPSAyLApwcm9wVCA9IHt9LApzdGFja0RlcHRoID0gMCwKc3RhdHVzID0gImFjdGl2ZSIsCnVzZXJOYW1lID0gInpzaCIsCndWID0gIjAwMDAwMDAwNS4wMDAwMDAwMDguKnpmaW5hbCIsCn0sCn0sCm1wYXRoQSA9IHsKIi9zb2Z0d2FyZS9rcDkyMC1STDgvU3RhZ2VzLzIwMjFhL21vZHVsZXMvYWxsL01QSS9HQ0MvOS4zLjAvT3Blbk1QSS80LjEu
    RANK_TABLE_FILE=/home/snassyr/ascend_rank_table.json
    ASCEND_DEVICE_ID=0
    RANK_ID=0
    RANK_SIZE=8
    JOB_ID=10088
    EBDEVELLIBPFM=/software/kp920-RL8/Stages/2021a/software/libpfm/4.11.1-f6500e77-GCCcore-9.3.0/easybuild/Compiler-GCCcore-9.3.0-libpfm-4.11.1-f6500e77-easybuild-devel
    EBDEVELPAPI=/software/kp920-RL8/Stages/2021a/software/PAPI/6.0.0.1-70887df7-GCCcore-9.3.0/easybuild/Compiler-GCCcore-9.3.0-PAPI-6.0.0.1-70887df7-easybuild-devel
    EBROOTLIBPFM=/software/kp920-RL8/Stages/2021a/software/libpfm/4.11.1-f6500e77-GCCcore-9.3.0
    EBROOTPAPI=/software/kp920-RL8/Stages/2021a/software/PAPI/6.0.0.1-70887df7-GCCcore-9.3.0
    EBVERSIONLIBPFM=4.11.1-f6500e77
    EBVERSIONPAPI=6.0.0.1-70887df7
    _ModuleTable047_=MiIsICIvc29mdHdhcmUva3A5MjAtUkw4L1N0YWdlcy8yMDIxYS9tb2R1bGVzL2FsbC9Db21waWxlci9tcGkvR0NDLzkuMy4wIgosICIvc29mdHdhcmUva3A5MjAtUkw4L1N0YWdlcy8yMDIxYS9tb2R1bGVzL2FsbC9Db21waWxlci9HQ0MvOS4zLjAiLCAiL3NvZnR3YXJlL2twOTIwLVJMOC9TdGFnZXMvMjAyMWEvbW9kdWxlcy9hbGwvQ29tcGlsZXIvR0NDY29yZS85LjMuMCIsICIvc29mdHdhcmUva3A5MjAtUkw4L1N0YWdlcy8yMDIxYS9VSS9Db21waWxlcnMiCiwgIi9zb2Z0d2FyZS9rcDkyMC1STDgvU3RhZ2VzLzIwMjFhL1VJL1Rvb2xzIiwgIi9zb2Z0d2FyZS9tb2R1bGVzL3N5c3RlbSIsICIvc29mdHdhcmUvbW9kdWxlcy9jdXN0b20iLAp9LApzeXN0ZW1CYXNlTVBBVEgg
    _ModuleTable048_=PSAiL29wdC9vaHBjL3B1Yi9tb2R1bGVmaWxlcyIsCn0K
