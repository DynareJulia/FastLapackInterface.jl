module BenchCholesky
using BenchmarkTools
using FastLapackInterface
using LinearAlgebra, LinearAlgebra.LAPACK

const sizes = (3, 4, 5, 10, 50, 100)
const vector_length = 100

const suite = BenchmarkGroup()

#### Cholesky

suite["Cholesky"] = BenchmarkGroup()
suite["Cholesky"]["creation"] = BenchmarkGroup()

for n in sizes
    A = rand(n, n)
    suite["Cholesky"]["creation"]["$n"] = @benchmarkable CholeskyPivotedWs($A)
end
suite["Cholesky"]["pstrf!"] = BenchmarkGroup()
suite["Cholesky"]["pstrf!"]["workspace"] = BenchmarkGroup()
suite["Cholesky"]["pstrf!"]["LAPACK"] = BenchmarkGroup()

function bench_pstrf!(As, ws)
    for A in As
        LAPACK.pstrf!(ws, 'U', A, 1e-6)
    end
end
function bench_pstrf!(As)
    for A in As
        LAPACK.pstrf!('U', A, 1e-6)
    end
end

for n in sizes
    As = [rand(n, n) for i in 1:vector_length]
    ws = CholeskyPivotedWs(As[1])
    suite["Cholesky"]["pstrf!"]["workspace"]["$n"] = @benchmarkable bench_pstrf!($As, $ws)
    suite["Cholesky"]["pstrf!"]["LAPACK"]["$n"] = @benchmarkable bench_pstrf!($As)
end

end
BenchCholesky.suite
