using Test
using FastLapackInterface
using LinearAlgebra.LAPACK

@testset "EigenWs" begin
    n = 3
    @testset "Real, square" begin
        A0 = randn(n, n)
        ws = EigenWs(copy(A0))

        A1, t1, s1, x1 = LAPACK.geev!('N', 'V', copy(A0))
        A2, t2, s2, x2 = LAPACK.geev!('N', 'V', copy(A0), ws)
        @test isapprox(A1, A2)
        @test isapprox(t1, t2)
        @test isapprox(s1, s2)
        @test isapprox(x1, x2)
        show(devnull, "text/plain", ws)
    end

    @testset "Complex, square" begin
        A0 = randn(ComplexF64, n, n)
        ws = EigenWs(copy(A0))

        W1, vs1, wr1 = LAPACK.geev!('N', 'V', copy(A0))
        W2, vs2, wr2 = LAPACK.geev!('N', 'V', copy(A0), ws)
        @test isapprox(W1, W2)
        @test isapprox(vs1, vs2)
        @test isapprox(wr1, wr2)
        show(devnull, "text/plain", ws)
    end
end
