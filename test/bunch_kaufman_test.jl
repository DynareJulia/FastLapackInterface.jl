using Test
using FastLapackInterface
using LinearAlgebra.LAPACK

@testset "BunchKaufmanWs" begin
    n = 3
    for T in (Float32, Float64, ComplexF32, ComplexF64)
        @testset "$(T) sytrf!" begin
            A = rand(T, n, n)
            ws = BunchKaufmanWs(copy(A))
            
            A1, ipiv1, inf1 = LAPACK.sytrf!(ws, 'U', copy(A)) 
            A2, ipiv2, inf2 = LAPACK.sytrf!('U', copy(A))
            @test A1 == A2
            @test ipiv1 == ipiv2

            ws = Workspace(LAPACK.sytrf!, copy(A))
            A1, ipiv1, inf1 = LAPACK.sytrf!(ws, 'U', copy(A)) 
            @test A1 == A2
            @test ipiv1 == ipiv2
        end
        @testset "$(T) sytrf_rook!" begin
            A = rand(T, n, n)
            ws = BunchKaufmanWs(copy(A))
            
            A1, ipiv1, inf1 = LAPACK.sytrf_rook!(ws, 'U', copy(A)) 
            A2, ipiv2, inf2 = LAPACK.sytrf_rook!('U', copy(A))
            @test A1 == A2
            @test ipiv1 == ipiv2
        end
    end
end
