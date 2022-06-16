module BenchQR
using BenchmarkTools
using FastLapackInterface
using LinearAlgebra, LinearAlgebra.LAPACK

const sizes = (3, 4, 5, 10, 50, 100)
const vector_length = 10

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

suite["QRWYWs"] = BenchmarkGroup()
suite["QRWYWs"]["creation"] = BenchmarkGroup()
for n in sizes
    suite["QRWYWs"]["creation"]["$n"] = @benchmarkable QRWYWs(A) setup=(A = rand($n, $n))
end

suite["QRpWs"] = BenchmarkGroup()
suite["QRpWs"]["creation"] = BenchmarkGroup()
for n in sizes
    suite["QRpWs"]["creation"]["$n"] = @benchmarkable QRWYWs(A) setup=(A = rand($n, $n))
end

end
BenchQR.suite
