using Test
using FastLapackInterface
using LinearAlgebra.LAPACK

@testset "QRWs" begin
    n = 3
    @testset "Real, square" begin
        A0 = randn(n, n)
        ws = QRWs(A0)
        @testset "geqrf!" begin
            A = copy(A0)
            AT, tau = LAPACK.geqrf!(ws, A)

            AT1, tau1 = LAPACK.geqrf!(copy(A0), randn(n))
            @test AT1 == AT
            @test tau1 == tau
        end

        @testset "ormqr!" begin
            C = randn(n, n)
            tau = randn(n)
            ws.τ .= tau
            Cout = LAPACK.ormqr!('L', 'N', copy(A0), tau, copy(C))
            Cout2 = LAPACK.ormqr!(ws, 'L', 'N', copy(A0), copy(C))
            @test isapprox(Cout, Cout2)

            # Is more testing required?
            Cout2 = LAPACK.ormqr!(ws, 'L', 'T', copy(A0)', copy(C))
            @test isapprox(Cout, Cout2)
        end
        show(devnull, "text/plain", ws)
    end
end

@testset "QRWYWs" begin
    n = 3
    @testset "Real, square" begin
        A0 = randn(n, n)

        @testset "geqrt!" begin
            A = copy(A0)
            ws = QRWYWs(A)
            AT, taut = LAPACK.geqrt!(ws, A)

            AT1, taut1 = LAPACK.geqrt!(copy(A0), zeros(size(ws.T)))
            @test AT1 == AT
            @test isapprox(taut1, taut)
            show(devnull, "text/plain", ws)
        end
    end
end

@testset "QRpWs" begin
    n = 3
    @testset "Real, square" begin
        A0 = randn(n, n)

        @testset "geqp3!" begin
            A = copy(A0)
            ws = QRpWs(A)
            AT, taut, jpvt = LAPACK.geqp3!(ws, A)

            AT1, taut1, jpvt1 = LAPACK.geqp3!(copy(A0), zeros(Int, length(ws.jpvt)),
                                              zeros(size(ws.τ)))
            @test isapprox(AT1, AT)
            @test isapprox(jpvt1, jpvt)
            @test isapprox(taut1, taut)
            show(devnull, "text/plain", ws)
        end
    end
end
