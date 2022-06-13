using Test
using FastLapackInterface
using LinearAlgebra.LAPACK
# A = diagm([1, -0.5, 1])

# ws = DgeesWs(3)

# dgees!(ws, A)
# println(ws.eigen_values)

# dgees!(ws, A, >=, 1.0)
# println(ws.eigen_values)

@testset "GeesWs" begin
    n = 3
    @testset "Real, square" begin
        A0 = randn(n, n)
        ws = FastLapackInterface.GeesWs(copy(A0))

        A1, vs1, wr1 = LAPACK.gees!('V', copy(A0))
        A2, vs2, wr2 = FastLapackInterface.gees!('V', copy(A0), ws)
        @test isapprox(A1, A2)
        @test isapprox(vs1, vs2)
        @test isapprox(wr1, wr2)
    end
    @testset "Real, square, select" begin
        A0 = [ 0.689816  0.173898  -0.489104
              -1.48437   1.06514    2.19973
              -0.239769  1.57603    0.330085]
              
        ws1 = FastLapackInterface.GeesWs(copy(A0))
        A1, vs1, wr1 = FastLapackInterface.gees!((wr, wi) -> wi^2 + wr^2 >= 1, 'V', copy(A0), ws1)
        @test ws1.sdim[] == 2
        
        ws2 = FastLapackInterface.GeesWs(copy(A0))
        A2, vs2, wr2 = FastLapackInterface.gees!('V', copy(A0), ws2)
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
        ws = FastLapackInterface.GgesWs(copy(A0))

        A1, B1, eig1, β1, vsl1, vsr1 = LAPACK.gges!('V', 'V', copy(A0), copy(B0))
        A2, B2, eig2, β2, vsl2, vsr2 = FastLapackInterface.gges!('V', 'V', copy(A0), copy(B0), ws)
        @test isapprox(A1, A2)
        @test isapprox(B1, B2)
        @test isapprox(eig1, eig2)
        @test isapprox(β1, β2)
        @test isapprox(vsl1, vsl2)
        @test isapprox(vsr1, vsr2)
    end
end
