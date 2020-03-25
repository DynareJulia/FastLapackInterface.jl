include("../src/LinSolveAlgo.jl")

using LinearAlgebra
using Test

n = 3
m = 2

for elty in (Float32, Float64, ComplexF32, ComplexF64)

    A0 = randn(elty, n, n)
    B0 = randn(elty, n, m)
    B1 = randn(elty, n, m)

    A = copy(A0)

    linws = LinSolveAlgo.LinSolveWs{elty, Int64}(n)

    LinSolveAlgo.lu!(A, linws)
    @test A == A0
    F = lu(A)
    @test UpperTriangular(linws.lu) ≈ F.U 

    LinSolveAlgo.lu!(A', linws)
    @test UpperTriangular(linws.lu) ≈ F.U

    B = copy(B0)
    LinSolveAlgo.linsolve_core!(A, B, linws)
    @test A == A0
    @test B ≈ A\B0

    copy!(B, B1)
    LinSolveAlgo.linsolve_core_no_lu!(A, B, linws)
    @test A == A0
    @test B ≈ A\B1

    copy!(B, B0)
    LinSolveAlgo.linsolve_core!(A', B, linws)
    @test B ≈ A'\B0

    copy!(B, B1)
    LinSolveAlgo.linsolve_core_no_lu!(A', B, linws)
    @test A == A0
    @test B ≈ A'\B1

    
    linws1 = LinSolveAlgo.LinSolveWs{elty, Int64}(n-1)
    C = view(A, 1:n-1, 1:n-1)
    C1 = copy(C)
    LinSolveAlgo.lu!(C1, linws1)
    F = lu!(C)
    @test triu(linws1.lu) ≈ F.U
    @test tril(linws1.lu, -1) ≈ tril(F.L, -1)
    
    D0 = view(B0, 1:n-1, 1:m-1)
    D1 = view(B1, 1:n-1, 1:m-1)
    D = copy(D0)
    LinSolveAlgo.linsolve_core!(C, D, linws1)
    @test C == view(A, 1:n-1, 1:n-1)
    @test D ≈ C\D0

    D = copy(D1)
    LinSolveAlgo.linsolve_core_no_lu!(C, D, linws1)
    @test C == view(A, 1:n-1, 1:n-1)
    @test D ≈ C\D1

    D = copy(D0)
    LinSolveAlgo.linsolve_core!(C', D, linws1)
    @test C == view(A, 1:n-1, 1:n-1)
    @test D ≈ C'\D0

    D = copy(D1)
    LinSolveAlgo.linsolve_core_no_lu!(C', D, linws1)
    @test C == view(A, 1:n-1, 1:n-1)
    @test D ≈ C'\D1
end
