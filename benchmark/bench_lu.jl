module BenchLU
using BenchmarkTools
using FastLapackInterface
using LinearAlgebra, LinearAlgebra.LAPACK

const sizes = (3, 4, 5, 10, 50, 100)
const vector_length = 100

const suite = BenchmarkGroup()

#### LUWs

suite["LUWs"] = BenchmarkGroup()
suite["LUWs"]["creation"] = BenchmarkGroup()

for n in sizes
    A = rand(n, n)
    suite["LUWs"]["creation"]["$n"] = @benchmarkable LUWs($A)
end
suite["LUWs"]["getrf!"]              = BenchmarkGroup()
suite["LUWs"]["getrf!"]["workspace"] = BenchmarkGroup()
suite["LUWs"]["getrf!"]["LAPACK"]    = BenchmarkGroup()

function bench_getrf!(As, ws)
    for A in As
        LAPACK.getrf!(A, ws)
    end
end
function bench_getrf!(As)
    for A in As
        LAPACK.getrf!(A)
    end
end

for n in sizes
    As = [rand(n, n) for i in 1:vector_length]
    ws = LUWs(As[1])
    suite["LUWs"]["getrf!"]["workspace"]["$n"] = @benchmarkable bench_getrf!($As, $ws)
    suite["LUWs"]["getrf!"]["LAPACK"]["$n"] = @benchmarkable bench_getrf!($As)
end

end
BenchLU.suite
