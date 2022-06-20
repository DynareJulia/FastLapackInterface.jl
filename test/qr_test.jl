using Test
using FastLapackInterface
using LinearAlgebra.LAPACK

#for elty in (Float32, Float64, ComplexF32, ComplexF64)

#    A0 = randn(elty, n, n)

# @static if VERSION < v"1.7"
#     target = qr(A0, Val(true))
# else
#     target = qr(A0, ColumnNorm())
# end

@testset "QRWs" begin
    n = 3
    @testset "Real, square" begin
        A0 = randn(n, n)
        ws = QRWs(A0)
        @testset "geqrf!" begin
            A = copy(A0)
            AT, tau = LAPACK.geqrf!(A, ws)

            AT1, tau1 = LAPACK.geqrf!(copy(A0), randn(n))
            @test AT1 == AT
            @test tau1 == tau
        end

        @testset "ormqr!" begin
            C = randn(n, n)
            tau = randn(n)
            ws.τ .= tau
            Cout = LAPACK.ormqr!('L', 'N', copy(A0), tau, copy(C))
            Cout2 = LAPACK.ormqr!('L', 'N', copy(A0), copy(C), ws)
            @test isapprox(Cout, Cout2)

            # Is more testing required?
            Cout2 = LAPACK.ormqr!('L', 'T', copy(A0)', copy(C), ws)
            @test isapprox(Cout, Cout2)
        end
        # redirect_stdout(devnull) do
        show(ws)
        # end
    end
end

@testset "QRWYWs" begin
    n = 3
    @testset "Real, square" begin
        A0 = randn(n, n)

        @testset "geqrt!" begin
            A = copy(A0)
            ws = QRWYWs(A)
            AT, taut = LAPACK.geqrt!(A, ws)

            AT1, taut1 = LAPACK.geqrt!(copy(A0), zeros(size(ws.T)))
            @test AT1 == AT
            @test isapprox(taut1, taut)
            # redirect_stdout(devnull) do
            show(ws)
            # end
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
            AT, taut, jpvt = LAPACK.geqp3!(A, ws)

            AT1, taut1, jpvt1 = LAPACK.geqp3!(copy(A0), zeros(Int, length(ws.jpvt)),
                                              zeros(size(ws.τ)))
            @test isapprox(AT1, AT)
            @test isapprox(jpvt1, jpvt)
            @test isapprox(taut1, taut)
            # redirect_stdout(devnull) do
            show(ws)
            # end
        end
    end
end
