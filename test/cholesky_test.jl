using Test
using FastLapackInterface
using LinearAlgebra.LAPACK

@testset "CholeskyPivotedWs" begin
    n = 3
    for T in (Float32, Float64, ComplexF32, ComplexF64)
        @testset "$(T) pstrf!" begin
            A = T[1 0 0;0 3.1 0.1; 2.0 0.0 5.0]
            A = (A .+ A') ./ 2
            ws = CholeskyPivotedWs(copy(A))
            
            A1, ipiv1, rank1, inf1 = LAPACK.pstrf!('U', copy(A), 1e-16) 
            A2, ipiv2, rank2, inf2 = LAPACK.pstrf!(ws, 'U', copy(A), 1e-16)
            @test A1 == A2
            @test ipiv1 == ipiv2
            @test rank1 == rank2

            ws = Workspace(LAPACK.pstrf!, copy(A))
            A1, ipiv1, rank1, inf1 = decompose!(ws, Hermitian(copy(A)), 1e-16)
            @test A1 == A2
            @test ipiv1 == ipiv2
            @test rank1 == rank2
            
            A1, ipiv1, rank1, inf1 = decompose!(ws, 'U', copy(A), 1e-16)
            @test A1 == A2
            @test ipiv1 == ipiv2
            @test rank1 == rank2

            decompose!(ws,'U', rand(n+1, n+1), 1e-16)
            @test length(ws.piv) == n+1
        end
    end
end
