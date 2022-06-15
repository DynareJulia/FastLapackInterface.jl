using Test
using FastLapackInterface
using LinearAlgebra.LAPACK
@testset "GeesWs" begin
    n = 3
    @testset "Real, square" begin
        A0 = randn(n, n)
        ws = GeesWs(copy(A0))

        A1, vs1, wr1 = LAPACK.gees!('V', copy(A0))
        A2, vs2, wr2 = LAPACK.gees!('V', copy(A0), ws)
        @test isapprox(A1, A2)
        @test isapprox(vs1, vs2)
        @test isapprox(wr1, wr2)
    end

    #TODO: This should be tested with something realistic
    @testset "Real, square, select" begin
        A0 = [0.689816 0.173898 -0.489104
              -1.48437 1.06514 2.19973
              -0.239769 1.57603 0.330085]

        ws1 = GeesWs(copy(A0))
        A1, vs1, wr1 = LAPACK.gees!((wr, wi) -> wi^2 + wr^2 >= 1, 'V',
                                    copy(A0), ws1)
        @test ws1.sdim[] == 2

        ws2 = GeesWs(copy(A0))
        A2, vs2, wr2 = LAPACK.gees!('V', copy(A0), ws2)
        @test wr1[1] == wr2[1]
        @test wr1[2] == wr2[3]
        @test wr1[3] == wr2[2]
    end
end

@testset "GgesWs" begin
    n = 3
    @testset "Real, square" begin
        A0 = randn(n, n)
        B0 = randn(n, n)
        ws = GgesWs(copy(A0))

        A1, B1, eig1, β1, vsl1, vsr1 = LAPACK.gges!('V', 'V', copy(A0), copy(B0))
        A2, B2, eig2, β2, vsl2, vsr2 = LAPACK.gges!('V', 'V', copy(A0), copy(B0), ws)
        @test isapprox(A1, A2)
        @test isapprox(B1, B2)
        @test isapprox(eig1, eig2)
        @test isapprox(β1, β2)
        @test isapprox(vsl1, vsl2)
        @test isapprox(vsr1, vsr2)
    end

    #TODO: This should be tested with something realistic
    @testset "Real, square, select" begin
        A0 = [-1.1189 -1.1333 -0.985796
              1.32901 0.628691 0.651912
              0.924541 0.287144 -1.09629]

        B0 = [-0.256655 -0.626563 -0.948712
              0.00727555 0.701693 -0.498145
              0.86268 -0.212549 -0.211994]
        ws = GgesWs(copy(A0))

        A0, B0, eig, β, vsl, vsr = LAPACK.gges!((ar, ai, b) -> ar^2 + ai^2 <
                                                FastLapackInterface.SCHUR_CRITERIUM *
                                                b^2, 'V', 'V',
                                                copy(A0), copy(B0), ws)
        @test ws.sdim[] == 1
        @test sign(real(eig[1])) == -1
        @test sign(real(eig[2])) == 1
    end
end
