module BenchSchur
using BenchmarkTools
using FastLapackInterface
using LinearAlgebra, LinearAlgebra.LAPACK

const sizes = (3, 4, 5, 10, 50, 100)
const vector_length = 100

const suite = BenchmarkGroup()

#### SchurWs

suite["SchurWs"] = BenchmarkGroup()
suite["SchurWs"]["creation"] = BenchmarkGroup()

for n in sizes
    A = rand(n, n)
    suite["SchurWs"]["creation"]["$n"] = @benchmarkable SchurWs($A)
end

suite["SchurWs"]["gees!"]                     = BenchmarkGroup()
suite["SchurWs"]["gees!"]["workspace"]        = BenchmarkGroup()
suite["SchurWs"]["gees!"]["workspace_select"] = BenchmarkGroup()
suite["SchurWs"]["gees!"]["LAPACK"]           = BenchmarkGroup()

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
    ws = SchurWs(As[1])
    suite["SchurWs"]["gees!"]["workspace"]["$n"] = @benchmarkable bench_gees!($As, $ws)
    sel_func = FastLapackInterface.gees_default_select()
    suite["SchurWs"]["gees!"]["workspace_select"]["$n"] = @benchmarkable bench_gees!($sel_func, $As, $ws)
    suite["SchurWs"]["gees!"]["LAPACK"]["$n"] = @benchmarkable bench_gees!($As)
end

#### GeneralizedSchurWs

suite["GeneralizedSchurWs"] = BenchmarkGroup()
suite["GeneralizedSchurWs"]["creation"] = BenchmarkGroup()

for n in sizes
    A = rand(n, n)
    suite["GeneralizedSchurWs"]["creation"]["$n"] = @benchmarkable GeneralizedSchurWs($A)
end

suite["GeneralizedSchurWs"]["gges!"]                     = BenchmarkGroup()
suite["GeneralizedSchurWs"]["gges!"]["workspace"]        = BenchmarkGroup()
suite["GeneralizedSchurWs"]["gges!"]["workspace_select"] = BenchmarkGroup()
suite["GeneralizedSchurWs"]["gges!"]["LAPACK"]           = BenchmarkGroup()

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
    ws = GeneralizedSchurWs(As[1])
    suite["GeneralizedSchurWs"]["gges!"]["workspace"]["$n"] = @benchmarkable bench_gges!($As, $ws)
    sel_func = FastLapackInterface.gges_default_select()
    suite["GeneralizedSchurWs"]["gges!"]["workspace_select"]["$n"] = @benchmarkable bench_gges!($sel_func, $As, $ws)
    suite["GeneralizedSchurWs"]["gges!"]["LAPACK"]["$n"] = @benchmarkable bench_gges!($As)
end

end
BenchSchur.suite
