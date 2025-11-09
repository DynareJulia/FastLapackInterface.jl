module BenchQR
using BenchmarkTools
using FastLapackInterface
using LinearAlgebra, LinearAlgebra.LAPACK

const sizes = (3, 4, 5, 10, 50, 100)
const vector_length = 100

const suite = BenchmarkGroup()

suite["QRWs"] = BenchmarkGroup()
suite["QRWs"]["creation"] = BenchmarkGroup()
for n in sizes
    A = rand(n, n)
    suite["QRWs"]["creation"]["$n"] = @benchmarkable QRWs($A)
end
suite["QRWs"]["geqrf!"] = BenchmarkGroup()
suite["QRWs"]["geqrf!"]["workspace"] = BenchmarkGroup()
suite["QRWs"]["geqrf!"]["LAPACK"] = BenchmarkGroup()

function bench_geqrf!(As, ws)
    for A in As
        LAPACK.geqrf!(ws, A)
    end
end
function bench_geqrf!(As)
    for A in As
        LAPACK.geqrf!(A)
    end
end

for n in sizes
    As = [rand(n, n) for i in 1:vector_length]
    ws = QRWs(As[1])
    suite["QRWs"]["geqrf!"]["workspace"]["$n"] = @benchmarkable bench_geqrf!($As, $ws)
    suite["QRWs"]["geqrf!"]["LAPACK"]["$n"] = @benchmarkable bench_geqrf!($As)
end
suite["QRWs"]["ormqr!"] = BenchmarkGroup()
suite["QRWs"]["ormqr!"]["workspace"] = BenchmarkGroup()
suite["QRWs"]["ormqr!"]["LAPACK"] = BenchmarkGroup()
function bench_ormqr!(As, ws::QRWs)
    for A in As
        LAPACK.ormqr!(ws, 'L', 'N', A, A)
    end
end
function bench_ormqr!(As, τ::Vector)
    for A in As
        LAPACK.ormqr!('L', 'N', A, τ, A)
    end
end

for n in sizes
    As = [rand(n, n) for i in 1:vector_length]
    ws = QRWs(As[1])
    LAPACK.geqrf!(ws, As[1])
    τ = copy(ws.τ)
    suite["QRWs"]["ormqr!"]["workspace"]["$n"] = @benchmarkable bench_ormqr!($As, $ws)
    suite["QRWs"]["ormqr!"]["LAPACK"]["$n"] = @benchmarkable bench_ormqr!($As, $τ)
end

####### QRWYWs

suite["QRWYWs"] = BenchmarkGroup()
suite["QRWYWs"]["creation"] = BenchmarkGroup()
for n in sizes
    suite["QRWYWs"]["creation"]["$n"] = @benchmarkable QRWYWs(A) setup=(A = rand($n, $n))
end

suite["QRWYWs"]["geqrt!"] = BenchmarkGroup()
suite["QRWYWs"]["geqrt!"]["workspace"] = BenchmarkGroup()
suite["QRWYWs"]["geqrt!"]["LAPACK"] = BenchmarkGroup()

function bench_geqrt!(As, ws)
    for A in As
        LAPACK.geqrt!(ws, A)
    end
end
function bench_geqrt!(As, T::Matrix)
    for A in As
        LAPACK.geqrt!(A, T)
    end
end

for n in sizes
    As = [rand(n, n) for i in 1:vector_length]
    T = rand(n, n)
    ws = QRWYWs(As[1])

    suite["QRWYWs"]["geqrt!"]["workspace"]["$n"] = @benchmarkable bench_geqrt!($As, $ws)
    suite["QRWYWs"]["geqrt!"]["LAPACK"]["$n"] = @benchmarkable bench_geqrt!($As, $T)
end

###### QRPivotedWs

suite["QRPivotedWs"] = BenchmarkGroup()
suite["QRPivotedWs"]["creation"] = BenchmarkGroup()
for n in sizes
    suite["QRPivotedWs"]["creation"]["$n"] = @benchmarkable QRWYWs(A) setup=(A = rand(
        $n, $n))
end

for n in sizes
    suite["QRPivotedWs"]["creation"]["$n"] = @benchmarkable QRWYWs(A) setup=(A = rand(
        $n, $n))
end

suite["QRPivotedWs"]["geqp3!"] = BenchmarkGroup()
suite["QRPivotedWs"]["geqp3!"]["workspace"] = BenchmarkGroup()
suite["QRPivotedWs"]["geqp3!"]["LAPACK"] = BenchmarkGroup()

function bench_geqp3!(As, ws)
    for A in As
        LAPACK.geqp3!(ws, A)
    end
end
function bench_geqp3!(As, lpvt, τ)
    for A in As
        LAPACK.geqp3!(A, lpvt, τ)
    end
end

for n in sizes
    As = [rand(n, n) for i in 1:vector_length]
    τ = zeros(n)
    lpvt = zeros(Int, n)
    ws = QRPivotedWs(As[1])

    suite["QRPivotedWs"]["geqp3!"]["workspace"]["$n"] = @benchmarkable bench_geqp3!(
        $As, $ws)
    suite["QRPivotedWs"]["geqp3!"]["LAPACK"]["$n"] = @benchmarkable bench_geqp3!(
        $As, $lpvt, $τ)
end

end
BenchQR.suite
