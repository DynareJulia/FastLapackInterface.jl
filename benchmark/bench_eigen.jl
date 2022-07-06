module BenchEigen
    using BenchmarkTools
    using FastLapackInterface
    using LinearAlgebra, LinearAlgebra.LAPACK

    const sizes = (3, 4, 5, 10, 50, 100)
    const vector_length = 100

    const suite = BenchmarkGroup()

    suite["EigenWs"] = BenchmarkGroup()
    suite["EigenWs"]["creation"] = BenchmarkGroup()
    for n in sizes
        A = rand(n, n)
        suite["EigenWs"]["creation"]["$n"] = @benchmarkable EigenWs($A)
    end

    suite["EigenWs"]["geevx!"]              = BenchmarkGroup()
    suite["EigenWs"]["geevx!"]["workspace"] = BenchmarkGroup()
    suite["EigenWs"]["geevx!"]["LAPACK"]    = BenchmarkGroup()

    function bench_geevx!(As, ws)
        for A in As
            LAPACK.geevx!(ws, 'N', 'V', 'V', 'V', A)
        end
    end
    function bench_geevx!(As)
        for A in As
            LAPACK.geevx!('N', 'V', 'V', 'V', A)
        end
    end
    for n in sizes
        As = [rand(n, n) for i in 1:vector_length]
        ws = EigenWs(As[1], sense=true)
        suite["EigenWs"]["geevx!"]["workspace"]["$n"] = @benchmarkable bench_geevx!($As, $ws)
        suite["EigenWs"]["geevx!"]["LAPACK"]["$n"] = @benchmarkable bench_geevx!($As)
    end
    
    suite["HermitianEigenWs"] = BenchmarkGroup()
    suite["HermitianEigenWs"]["creation"] = BenchmarkGroup()
    for n in sizes
        A = rand(n, n)
        suite["HermitianEigenWs"]["creation"]["$n"] = @benchmarkable HermitianEigenWs($A)
    end

    suite["HermitianEigenWs"]["syevr!"]              = BenchmarkGroup()
    suite["HermitianEigenWs"]["syevr!"]["workspace"] = BenchmarkGroup()
    suite["HermitianEigenWs"]["syevr!"]["LAPACK"]    = BenchmarkGroup()

    function bench_syevr!(As, ws)
        for A in As
            LAPACK.syevr!(ws, 'V', 'A', 'U', A, 0.0, 0.0, 0, 0, 1e-6)
        end
    end
    function bench_syevr!(As)
        for A in As
            LAPACK.syevr!('V', 'A', 'U', A, 0.0, 0.0, 0, 0, 1e-6)
        end
    end
    for n in sizes
        As = [rand(ComplexF64, n, n) for i in 1:vector_length]
        As .= (As .+ adjoint.(As))./2
        
        ws = HermitianEigenWs(As[1], vecs=true)
        suite["HermitianEigenWs"]["syevr!"]["workspace"]["$n"] = @benchmarkable bench_syevr!($As, $ws)
        suite["HermitianEigenWs"]["syevr!"]["LAPACK"]["$n"] = @benchmarkable bench_syevr!($As)
    end
    
end
BenchEigen.suite
