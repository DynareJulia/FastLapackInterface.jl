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

suite["QRWs"]["geqrf!_workspace"] = BenchmarkGroup()
suite["QRWs"]["geqrf!_LAPACK"] = BenchmarkGroup()

function bench_geqrf!(As, ws)
    for A in As
        LAPACK.geqrf!(A, ws)
    end
end
function bench_geqrf!(As)
    for A in As
        LAPACK.geqrf!(A)
    end
end

for n in sizes
    As = [rand(n, n) for i=1:vector_length]
    ws = QRWs(As[1])
    suite["QRWs"]["geqrf!_workspace"]["$n"] = @benchmarkable bench_geqrf!($As, $ws)
    suite["QRWs"]["geqrf!_LAPACK"]["$n"] = @benchmarkable bench_geqrf!($As)
end

suite["QRWs"]["ormqr!_workspace"] = BenchmarkGroup()
suite["QRWs"]["ormqr!_LAPACK"] = BenchmarkGroup()
function bench_ormqr!(As, ws::QRWs)
    for A in As
        LAPACK.ormqr!('L', 'N', A, A, ws)
    end
end
function bench_ormqr!(As, τ::Vector)
    for A in As
        LAPACK.ormqr!('L', 'N', A, τ, A)
    end
end

for n in sizes
    As = [rand(n, n) for i=1:vector_length]
    τ  = zeros(n) 
    ws = QRWs(As[1])
    suite["QRWs"]["ormqr!_workspace"]["$n"] = @benchmarkable bench_ormqr!($As, $ws)
    suite["QRWs"]["ormqr!_LAPACK"]["$n"] = @benchmarkable bench_ormqr!($As, $τ)
end

####### QRWYWs

suite["QRWYWs"] = BenchmarkGroup()
suite["QRWYWs"]["creation"] = BenchmarkGroup()
for n in sizes
    suite["QRWYWs"]["creation"]["$n"] = @benchmarkable QRWYWs(A) setup=(A = rand($n, $n))
end

suite["QRWYWs"]["geqrt!_workspace"] = BenchmarkGroup()
suite["QRWYWs"]["geqrt!_LAPACK"] = BenchmarkGroup()

function bench_geqrt!(As, ws)
    for A in As
        LAPACK.geqrt!(A, ws)
    end
end
function bench_geqrt!(As, T::Matrix)
    for A in As
        LAPACK.geqrt!(A, T)
    end
end

for n in sizes
    As = [rand(n, n) for i=1:vector_length]
    T = rand(n, n)
    ws = QRWYWs(As[1])
    suite["QRWYWs"]["geqrt!_workspace"]["$n"] = @benchmarkable bench_geqrt!($As, $ws)
    suite["QRWYWs"]["geqrt!_LAPACK"]["$n"]    = @benchmarkable bench_geqrt!($As, $T)
end

###### QRpWs

suite["QRpWs"] = BenchmarkGroup()
suite["QRpWs"]["creation"] = BenchmarkGroup()
for n in sizes
    suite["QRpWs"]["creation"]["$n"] = @benchmarkable QRWYWs(A) setup=(A = rand($n, $n))
end

for n in sizes
    suite["QRpWs"]["creation"]["$n"] = @benchmarkable QRWYWs(A) setup=(A = rand($n, $n))
end

suite["QRpWs"]["geqp3!_workspace"] = BenchmarkGroup()
suite["QRpWs"]["geqp3!_LAPACK"] = BenchmarkGroup()

function bench_geqp3!(As, ws)
    for A in As
        LAPACK.geqp3!(A, ws)
    end
end
function bench_geqp3!(As, lpvt, τ)
    for A in As
        LAPACK.geqp3!(A, lpvt, τ)
    end
end

for n in sizes
    As = [rand(n, n) for i=1:vector_length]
    τ  = zeros(n)
    lpvt = zeros(Int,n)
    ws = QRpWs(As[1])
    suite["QRpWs"]["geqp3!_workspace"]["$n"] = @benchmarkable bench_geqp3!($As, $ws)
    suite["QRpWs"]["geqp3!_LAPACK"]["$n"]    = @benchmarkable bench_geqp3!($As, $lpvt, $τ )
end

end
BenchQR.suite
