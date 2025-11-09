module BenchBunchKaufman
using BenchmarkTools
using FastLapackInterface
using LinearAlgebra, LinearAlgebra.LAPACK

const sizes = (3, 4, 5, 10, 50, 100)
const vector_length = 100

const suite = BenchmarkGroup()

#### BunchKaufman

suite["BunchKaufman"] = BenchmarkGroup()
suite["BunchKaufman"]["creation"] = BenchmarkGroup()

for n in sizes
    A = rand(n, n)
    suite["BunchKaufman"]["creation"]["$n"] = @benchmarkable BunchKaufmanWs($A)
end
suite["BunchKaufman"]["sytrf!"] = BenchmarkGroup()
suite["BunchKaufman"]["sytrf!"]["workspace"] = BenchmarkGroup()
suite["BunchKaufman"]["sytrf!"]["LAPACK"] = BenchmarkGroup()

function bench_sytrf!(As, ws)
    for A in As
        LAPACK.sytrf!(ws, 'U', A)
    end
end
function bench_sytrf!(As)
    for A in As
        LAPACK.sytrf!('U', A)
    end
end

for n in sizes
    As = [rand(n, n) for i in 1:vector_length]
    ws = BunchKaufmanWs(As[1])
    suite["BunchKaufman"]["sytrf!"]["workspace"]["$n"] = @benchmarkable bench_sytrf!(
        $As, $ws)
    suite["BunchKaufman"]["sytrf!"]["LAPACK"]["$n"] = @benchmarkable bench_sytrf!($As)
end

suite["BunchKaufman"]["sytrf_rook!"] = BenchmarkGroup()
suite["BunchKaufman"]["sytrf_rook!"]["workspace"] = BenchmarkGroup()
suite["BunchKaufman"]["sytrf_rook!"]["LAPACK"] = BenchmarkGroup()

function bench_sytrf_rook!(As, ws)
    for A in As
        LAPACK.sytrf_rook!(ws, 'U', A)
    end
end
function bench_sytrf_rook!(As)
    for A in As
        LAPACK.sytrf_rook!('U', A)
    end
end

for n in sizes
    As = [rand(n, n) for i in 1:vector_length]
    ws = BunchKaufmanWs(As[1])
    suite["BunchKaufman"]["sytrf_rook!"]["workspace"]["$n"] = @benchmarkable bench_sytrf_rook!(
        $As, $ws)
    suite["BunchKaufman"]["sytrf_rook!"]["LAPACK"]["$n"] = @benchmarkable bench_sytrf_rook!($As)
end

suite["BunchKaufman"]["hetrf!"] = BenchmarkGroup()
suite["BunchKaufman"]["hetrf!"]["workspace"] = BenchmarkGroup()
suite["BunchKaufman"]["hetrf!"]["LAPACK"] = BenchmarkGroup()

function bench_hetrf!(As, ws)
    for A in As
        LAPACK.hetrf!(ws, 'U', A)
    end
end
function bench_hetrf!(As)
    for A in As
        LAPACK.hetrf!('U', A)
    end
end

for n in sizes
    As = [rand(ComplexF64, n, n) for i in 1:vector_length]
    ws = BunchKaufmanWs(As[1])
    suite["BunchKaufman"]["hetrf!"]["workspace"]["$n"] = @benchmarkable bench_hetrf!(
        $As, $ws)
    suite["BunchKaufman"]["hetrf!"]["LAPACK"]["$n"] = @benchmarkable bench_hetrf!($As)
end

suite["BunchKaufman"]["hetrf_rook!"] = BenchmarkGroup()
suite["BunchKaufman"]["hetrf_rook!"]["workspace"] = BenchmarkGroup()
suite["BunchKaufman"]["hetrf_rook!"]["LAPACK"] = BenchmarkGroup()

function bench_hetrf_rook!(As, ws)
    for A in As
        LAPACK.hetrf_rook!(ws, 'U', A)
    end
end
function bench_hetrf_rook!(As)
    for A in As
        LAPACK.hetrf_rook!('U', A)
    end
end

for n in sizes
    As = [rand(ComplexF64, n, n) for i in 1:vector_length]
    ws = BunchKaufmanWs(As[1])
    suite["BunchKaufman"]["hetrf_rook!"]["workspace"]["$n"] = @benchmarkable bench_hetrf_rook!(
        $As, $ws)
    suite["BunchKaufman"]["hetrf_rook!"]["LAPACK"]["$n"] = @benchmarkable bench_hetrf_rook!($As)
end

end
BenchBunchKaufman.suite
