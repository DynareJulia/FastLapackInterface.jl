module BenchSchur
using BenchmarkTools
using FastLapackInterface
using LinearAlgebra, LinearAlgebra.LAPACK

const sizes = (3, 4, 5, 10, 50, 100)
const vector_length = 100

const suite = BenchmarkGroup()

#### GeesWs

suite["GeesWs"] = BenchmarkGroup()
suite["GeesWs"]["creation"] = BenchmarkGroup()

for n in sizes
    A = rand(n, n)
    suite["GeesWs"]["creation"]["$n"] = @benchmarkable GeesWs($A)
end

suite["GeesWs"]["gees!"]                     = BenchmarkGroup()
suite["GeesWs"]["gees!"]["workspace"]        = BenchmarkGroup()
suite["GeesWs"]["gees!"]["workspace_select"] = BenchmarkGroup()
suite["GeesWs"]["gees!"]["LAPACK"]           = BenchmarkGroup()

function bench_gees!(As, ws)
    for A in As
        LAPACK.gees!('V', A, ws)
    end
end
function bench_gees!(sel_func, As, ws)
    for A in As
        LAPACK.gees!(sel_func, 'V', A, ws)
    end
end
function bench_gees!(As)
    for A in As
        LAPACK.gees!('V', A)
    end
end

for n in sizes
    As = [rand(n, n) for i=1:vector_length]
    ws = GeesWs(As[1])
    suite["GeesWs"]["gees!"]["workspace"]["$n"] = @benchmarkable bench_gees!($As, $ws)
    sel_func = FastLapackInterface.gees_default_select()
    suite["GeesWs"]["gees!"]["workspace_select"]["$n"] = @benchmarkable bench_gees!($sel_func, $As, $ws)
    suite["GeesWs"]["gees!"]["LAPACK"]["$n"] = @benchmarkable bench_gees!($As)
end

#### GgesWs

suite["GgesWs"] = BenchmarkGroup()
suite["GgesWs"]["creation"] = BenchmarkGroup()

for n in sizes
    A = rand(n, n)
    suite["GgesWs"]["creation"]["$n"] = @benchmarkable GgesWs($A)
end

suite["GgesWs"]["gges!"]                     = BenchmarkGroup()
suite["GgesWs"]["gges!"]["workspace"]        = BenchmarkGroup()
suite["GgesWs"]["gges!"]["workspace_select"] = BenchmarkGroup()
suite["GgesWs"]["gges!"]["LAPACK"]           = BenchmarkGroup()

function bench_gges!(As, ws)
    for A in As
        LAPACK.gges!('V', 'V', A, A, ws)
    end
end
function bench_gges!(sel_func, As, ws)
    for A in As
        LAPACK.gges!(sel_func, 'V', 'V', A, A, ws)
    end
end
function bench_gges!(As)
    for A in As
        LAPACK.gges!('V','V', A, A)
    end
end

for n in sizes
    As = [rand(n, n) for i=1:vector_length]
    ws = GgesWs(As[1])
    suite["GgesWs"]["gges!"]["workspace"]["$n"] = @benchmarkable bench_gges!($As, $ws)
    sel_func = FastLapackInterface.gges_default_select()
    suite["GgesWs"]["gges!"]["workspace_select"]["$n"] = @benchmarkable bench_gges!($sel_func, $As, $ws)
    suite["GgesWs"]["gges!"]["LAPACK"]["$n"] = @benchmarkable bench_gges!($As)
end

end
BenchSchur.suite
