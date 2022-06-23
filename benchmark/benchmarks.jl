using BenchmarkTools

using LinearAlgebra
LinearAlgebra.BLAS.set_num_threads(1)

const SUITE = BenchmarkGroup()
for file in sort(readdir(@__DIR__))
    if startswith(file, "bench_") && endswith(file, ".jl")
        SUITE[chop(file; head = length("bench_"), tail = length(".jl"))] = include(file)
    end
end
