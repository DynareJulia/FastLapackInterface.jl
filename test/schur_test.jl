using Random
using Test
using FastLapackInterface
using LinearAlgebra


function make_matrix(n)
    Random.seed!(123)
    A = randn(n, n)
    FA = eigen(A)
    k = findall(abs.(FA.values) .> 1.0)
    FA.values[k] .= 1.0
    B = real.(FA.vectors*diagm(FA.values)*inv(FA.vectors))
    FB = schur(B)
    return B, sum(abs.(FB.values) .> 1.0)
end

@testset "SchurWs" begin
    n = 10
    @testset "Real, square" begin
        A0 = randn(n, n)
        ws = SchurWs(copy(A0))

        A1, vs1, wr1 = LAPACK.gees!('V', copy(A0))
        A2, vs2, wr2 = LAPACK.gees!(ws, 'V', copy(A0))
        @test isapprox(A1, A2)
        @test isapprox(vs1, vs2)
        @test isapprox(wr1, wr2)
        # using Workspace, factorize!
        ws = Workspace(LAPACK.gees!, copy(A0))
        A2, vs2, wr2 = factorize!(ws, 'V', copy(A0))
        @test isapprox(A1, A2)
        @test isapprox(vs1, vs2)
        @test isapprox(wr1, wr2)
        show(devnull, "text/plain", ws)
        for div in (-1,1)
            @test_throws FastLapackInterface.WorkspaceSizeError factorize!(ws, 'V', randn(n+div, n+div); resize=false)
            factorize!(ws, 'V', randn(n+div, n+div))
            @test length(ws.wr) == n+div
        end
        
    end

    @testset "Real, square, select" begin
        A0 = [0.689816 0.173898 -0.489104
              -1.48437 1.06514 2.19973
              -0.239769 1.57603 0.330085]

        ws1 = SchurWs(copy(A0))
        A1, vs1, wr1 = LAPACK.gees!(ws1, 'V', copy(A0);
                                    select = (wr, wi) -> wi^2 + wr^2 >= 1)
        @test ws1.sdim[] == 2

        ws2 = SchurWs(copy(A0))
        A2, vs2, wr2 = LAPACK.gees!(ws2, 'V', copy(A0))
        @test wr1[1] == wr2[1]
        @test wr1[2] == wr2[3]
        @test wr1[3] == wr2[2]
    end
    @testset "Select keywords" begin
        n = 10
        A0, nunstable = make_matrix(n)
        ws = SchurWs(copy(A0))
        LAPACK.gees!(ws, 'V', copy(A0), select = FastLapackInterface.lhpm)
        @test count(abs.(ws.eigen_values) .< -FastLapackInterface.SCHUR_EPSILON) == ws.sdim[]
        LAPACK.gees!(ws, 'V', copy(A0), select = FastLapackInterface.lhpp)
        @test count(abs.(ws.eigen_values) .< FastLapackInterface.SCHUR_EPSILON) == ws.sdim[]
        LAPACK.gees!(ws, 'V', copy(A0), select = FastLapackInterface.rhpm)
        @test count(abs.(ws.eigen_values) .>=-FastLapackInterface.SCHUR_EPSILON) == ws.sdim[]
        LAPACK.gees!(ws, 'V', copy(A0), select = FastLapackInterface.rhpp)
        @test count(abs.(ws.eigen_values) .>= FastLapackInterface.SCHUR_EPSILON) == ws.sdim[]
        LAPACK.gees!(ws, 'V', copy(A0), select = FastLapackInterface.udim)
        @test count(abs.(ws.eigen_values) .< 1 - FastLapackInterface.SCHUR_EPSILON) == ws.sdim[]
        LAPACK.gees!(ws, 'V', copy(A0), select = FastLapackInterface.udip)
        @test count(abs.(ws.eigen_values) .< 1 + FastLapackInterface.SCHUR_EPSILON) == ws.sdim[]
        LAPACK.gees!(ws, 'V', copy(A0), select = FastLapackInterface.udom)
        @test count(abs.(ws.eigen_values) .>= 1 - FastLapackInterface.SCHUR_EPSILON) == ws.sdim[]
        LAPACK.gees!(ws, 'V', copy(A0), select = FastLapackInterface.udop)
        @test count(abs.(ws.eigen_values) .>= 1 + FastLapackInterface.SCHUR_EPSILON) == ws.sdim[]
    end
end

@testset "GeneralizedSchurWs" begin
    n = 10
    @testset "Real, square" begin
        A0 = randn(n, n)
        B0 = randn(n, n)
        ws = GeneralizedSchurWs(copy(A0))

        A1, B1, α1, β1, vsl1, vsr1 = LAPACK.gges!('V', 'V', copy(A0), copy(B0))
        A2, B2, α2, β2, vsl2, vsr2 = LAPACK.gges!(ws, 'V', 'V', copy(A0), copy(B0))
        @test isapprox(A1, A2)
        @test isapprox(B1, B2)
        @test isapprox(α1, α2)
        @test isapprox(β1, β2)
        @test isapprox(vsl1, vsl2)
        @test isapprox(vsr1, vsr2)

        # Using Workspace, factorize!
        ws = Workspace(LAPACK.gges!, copy(A0))
        A2, B2, α2, β2, vsl2, vsr2 = factorize!(ws, 'V', 'V', copy(A0), copy(B0))
        @test isapprox(A1, A2)
        @test isapprox(B1, B2)
        @test isapprox(α1, α2)
        @test isapprox(β1, β2)
        @test isapprox(vsl1, vsl2)
        @test isapprox(vsr1, vsr2)
        @test isapprox(sort(abs.(eigen(A0, B0).values)), sort(abs.(ws.eigen_values)))

        show(devnull, "text/plain", ws)

        for div in (-1,1)
            @test_throws FastLapackInterface.WorkspaceSizeError factorize!(ws, 'V', 'V', randn(n+div, n+div), randn(n+div, n+div); resize=false)
            factorize!(ws, 'V', 'V', randn(n+div, n+div), randn(n+div, n+div))
            @test length(ws.αr) == n+div
        end
    end

    #TODO: This should be tested with something realistic
    @testset "Real, square, select" begin
        A0 = [-1.1189 -1.1333 -0.985796
              1.32901 0.628691 0.651912
              0.924541 0.287144 -1.09629]

        B0 = [-0.256655 -0.626563 -0.948712
              0.00727555 0.701693 -0.498145
              0.86268 -0.212549 -0.211994]
        ws = GeneralizedSchurWs(copy(A0))

        A0, B0, α, β, vsl, vsr = LAPACK.gges!(ws, 'V', 'V', copy(A0),
                                                copy(B0);
                                                select = (ar, ai, b) -> ar^2 + ai^2 <
                                                                        FastLapackInterface.SCHUR_CRITERIUM *
                                                                        b^2)
        A0, B0, α, β, vsl, vsr = LAPACK.gges!(ws, 'V', 'V', copy(A0),
                                                copy(B0);
                                                select = FastLapackInterface.udim)
        @test ws.sdim[] == 1
        @test (real(α[1])/β[1])^2 < FastLapackInterface.SCHUR_CRITERIUM 
        @test (real(α[2])/β[2])^2 > FastLapackInterface.SCHUR_CRITERIUM 
        @test (real(α[3])/β[3])^2 > FastLapackInterface.SCHUR_CRITERIUM 
        @test isapprox(sort(abs.(eigen(A0, B0).values)), sort(abs.(ws.eigen_values)))
        show(devnull, "text/plain", ws)
    end

    @testset "Select keywords" begin
        n = 10
        A0, nunstable = make_matrix(n)
        ws = GeneralizedSchurWs(copy(A0))
        LAPACK.gges!(ws, 'V', 'V', copy(A0), Matrix(1.0*I(n));
                                            select = FastLapackInterface.lhpm)
        @test count(abs.(ws.eigen_values) .< -FastLapackInterface.SCHUR_EPSILON) == ws.sdim[]
        LAPACK.gges!(ws, 'V', 'V', copy(A0), Matrix(1.0*I(n));
                                            select = FastLapackInterface.lhpp)
        @test count(abs.(ws.eigen_values) .< FastLapackInterface.SCHUR_EPSILON) == ws.sdim[]
        LAPACK.gges!(ws, 'V', 'V', copy(A0), Matrix(1.0*I(n));
                                            select = FastLapackInterface.rhpm)
        @test count(abs.(ws.eigen_values) .>=-FastLapackInterface.SCHUR_EPSILON) == ws.sdim[]
        LAPACK.gges!(ws, 'V', 'V', copy(A0), Matrix(1.0*I(n));
                                            select = FastLapackInterface.rhpp)
        @test count(abs.(ws.eigen_values) .>= FastLapackInterface.SCHUR_EPSILON) == ws.sdim[]
        LAPACK.gges!(ws, 'V', 'V', copy(A0), Matrix(1.0*I(n));
                                            select = FastLapackInterface.udim)
        @test count(abs.(ws.eigen_values) .< 1 - FastLapackInterface.SCHUR_EPSILON) == ws.sdim[]
        LAPACK.gges!(ws, 'V', 'V', copy(A0), Matrix(1.0*I(n));
                                            select = FastLapackInterface.udip)
        @test count(abs.(ws.eigen_values) .< 1 + FastLapackInterface.SCHUR_EPSILON) == ws.sdim[]
        LAPACK.gges!(ws, 'V', 'V', copy(A0), Matrix(1.0*I(n));
                                            select = FastLapackInterface.udom)
        @test count(abs.(ws.eigen_values) .>= 1 - FastLapackInterface.SCHUR_EPSILON) == ws.sdim[]
        LAPACK.gges!(ws, 'V', 'V', copy(A0), Matrix(1.0*I(n));
                                            select = FastLapackInterface.udop)
        @test count(abs.(ws.eigen_values) .>= 1 + FastLapackInterface.SCHUR_EPSILON) == ws.sdim[]
    end
end
