using Test
using FastLapackInterface
using LinearAlgebra.LAPACK

@testset "LSEWs" begin
    n = 8
    m = 4
    p = 6 
    for T in (Float32, Float64, ComplexF32, ComplexF64)
        @testset "$T" begin
            A0 = rand(T, m, n)
            B0 = rand(T, p, n)
            c0 = rand(T, m)
            d0 = rand(T, p)
            ws = LSEWs(A0, B0)
            @testset "gglse!" begin
                A = copy(A0)
                B = copy(B0)
                c = copy(c0)
                d = copy(d0)
                X1, err1 = LAPACK.gglse!(copy(A0), copy(c0), copy(B0), copy(d0))
                X2, err2 = LAPACK.gglse!(ws, copy(A0), copy(c0), copy(B0), copy(d0))

                @test isapprox(X1, X2)
                @test isapprox(err1, err2)
                # using Workspace, factorize!
                ws = Workspace(LAPACK.gglse!, copy(A0))
                X2, err2 = factorize!(ws, copy(A0), copy(c0), copy(B0), copy(d0))
                
                @test isapprox(X1, X2)
                for div in (-1, 1)
                    @test_throws FastLapackInterface.WorkspaceSizeError factorize!(ws, rand(T, m, n+div), copy(c0), rand(T, p+div, n+div), rand(T, p+div); resize=false)
                    factorize!(ws, rand(T, m, n+div), copy(c0), rand(T, p+div, n+div), rand(T, p+div))
                    @test size(ws.X , 1) == n+div
                    @test size(ws.work , 1) >= p+div + min(m, n+div) + max(m,n+div)*32
                end
            end
        end
    end
end

