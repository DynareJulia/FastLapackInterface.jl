using Test
using FastLapackInterface
using LinearAlgebra.LAPACK

@testset "BunchKaufmanWs" begin
    n = 3
    for T in (Float32, Float64, ComplexF32, ComplexF64)
        @testset "$(T) sytrf!" begin
            A = rand(T, n, n)
            A = (A .+ transpose(A)) ./ 2
            ws = BunchKaufmanWs(copy(A))
            
            A1, ipiv1, inf1 = LAPACK.sytrf!(ws, 'U', copy(A)) 
            A2, ipiv2, inf2 = LAPACK.sytrf!('U', copy(A))
            @test A1 == A2
            @test ipiv1 == ipiv2

            ws = Workspace(LAPACK.sytrf!, copy(A))
            A1, ipiv1, inf1 = decompose!(ws, Symmetric(copy(A)))
            @test A1 == A2
            @test ipiv1 == ipiv2
            
            A1, ipiv1, inf1 = decompose!(ws, 'U', copy(A))
            @test A1 == A2
            @test ipiv1 == ipiv2
            for div in (-1,1)
                @test_throws FastLapackInterface.WorkspaceSizeError decompose!(ws, Symmetric(rand(T, n+div, n+div)); resize=false)
                decompose!(ws, Symmetric(rand(T, n+div, n+div)))
                @test length(ws.ipiv) == n+div
            end
        end
        @testset "$(T) sytrf_rook!" begin
            A = rand(T, n, n)
            A = (A .+ transpose(A))./2
            ws = BunchKaufmanWs(copy(A))
            
            A1, ipiv1, inf1 = LAPACK.sytrf_rook!(ws, 'U', copy(A)) 
            A2, ipiv2, inf2 = LAPACK.sytrf_rook!('U', copy(A))
            @test A1 == A2
            @test ipiv1 == ipiv2
            
            ws = Workspace(LAPACK.sytrf!, copy(A))
            A1, ipiv1, inf1 = decompose!(ws,  Symmetric(copy(A)); rook=true)
            @test A1 == A2
            @test ipiv1 == ipiv2
            
            A1, ipiv1, inf1 = decompose!(ws, 'U', copy(A); rook=true)
            @test A1 == A2
            @test ipiv1 == ipiv2
        end
    end
    for T in (ComplexF32, ComplexF64)
        @testset "$(T) hetrf!" begin
            A = rand(T, n, n)
            A = (A .+ A')./2
            ws = BunchKaufmanWs(copy(A))
            
            A1, ipiv1, inf1 = LAPACK.hetrf!(ws, 'U', copy(A)) 
            A2, ipiv2, inf2 = LAPACK.hetrf!('U', copy(A))
            @test A1 == A2
            @test ipiv1 == ipiv2

            ws = Workspace(LAPACK.hetrf!, copy(A))
            A1, ipiv1, inf1 = decompose!(ws, Hermitian(copy(A)))
            @test A1 == A2
            @test ipiv1 == ipiv2
            
            A1, ipiv1, inf1 = decompose!(ws, 'U', copy(A))
            @test A1 == A2
            @test ipiv1 == ipiv2
        end
        @testset "$(T) hetrf_rook!" begin
            A = rand(T, n, n)
            A = (A .+ A')./2
            ws = BunchKaufmanWs(copy(A))
            
            A1, ipiv1, inf1 = LAPACK.hetrf_rook!(ws, 'U', copy(A)) 
            A2, ipiv2, inf2 = LAPACK.hetrf_rook!('U', copy(A))
            @test A1 == A2
            @test ipiv1 == ipiv2
            
            ws = Workspace(LAPACK.hetrf!, copy(A))
            A1, ipiv1, inf1 = decompose!(ws,  Hermitian(copy(A)), rook=true)
            @test A1 == A2
            @test ipiv1 == ipiv2
            A1, ipiv1, inf1 = decompose!(ws, 'U', copy(A); rook=true)
            @test A1 == A2
            @test ipiv1 == ipiv2
        end
    end
end
