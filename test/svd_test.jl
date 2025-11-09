using Test
using FastLapackInterface
using LinearAlgebra.LAPACK

function test_sdd(m, n, numbertype)
    A1 = randn(numbertype, m, n)
    A2 = copy(A1)

    # default: job = 'A'
    ws = SVDsddWs(A2)

    U1, S1, VT1 = LAPACK.gesdd!('A', A1)
    U2, S2, VT2 = LAPACK.gesdd!(ws, 'A', A2)

    @test isapprox(U1, U2)
    @test isapprox(S1, S2)
    @test isapprox(VT1, VT2)

    # all possible jobs
    for j in ('A', 'S', 'O', 'N')
        A1 = randn(numbertype, m, n)
        A2 = copy(A1)

        ws = SVDsddWs(A2, job = j)

        U1, S1, VT1 = LAPACK.gesdd!(j, A1)
        U2, S2, VT2 = LAPACK.gesdd!(ws, j, A2)

        @test isapprox(U1, U2)
        @test isapprox(S1, S2)
        @test isapprox(VT1, VT2)
        if j == 'O'
            # A is overwritten with U
            @test isapprox(A1, A2)
            if m >= n
                @test isapprox(U2, A2)
            else
                @test isapprox(VT2, A2)
            end
        end
    end
end

function test_svd(m, n, numbertype)
    A1 = randn(numbertype, m, n)
    A2 = copy(A1)

    # default: jobu = jobvt = 'A'
    ws = SVDsvdWs(A2)

    U1, S1, VT1 = LAPACK.gesvd!('A', 'A', A1)
    U2, S2, VT2 = LAPACK.gesvd!(ws, 'A', 'A', A2)

    @test isapprox(U1, U2)
    @test isapprox(S1, S2)
    @test isapprox(VT1, VT2)

    # all possible jobs
    for j1 in ('A', 'S', 'O', 'N'), j2 in ('A', 'S', 'O', 'N')
        # j1 = j2 = 'O' can not happen
        j1 == 'O' && j2 == 'O' && continue
        A1 = randn(numbertype, m, n)
        A2 = copy(A1)

        ws = SVDsvdWs(A2, jobu = j1, jobvt = j2)

        U1, S1, VT1 = LAPACK.gesvd!(j1, j2, A1)
        U2, S2, VT2 = LAPACK.gesvd!(ws, j1, j2, A2)

        @test isapprox(U1, U2)
        @test isapprox(S1, S2)
        @test isapprox(VT1, VT2)
        if j1 == 'O'
            # A is overwritten with U
            @test isapprox(A1, A2)
            @test isapprox(U2, A2)
        elseif j2 == 'O'
            @test isapprox(A1, A2)
            @test isapprox(VT2, A2)
        end
    end
end

function test_generalizedsvd(m, n, numbertype)
    A1 = randn(numbertype, m, n)
    B1 = randn(numbertype, m, n)
    A2 = copy(A1)
    B2 = copy(B1)

    # default: jobu = 'U', jobv = 'V', jobq = 'Q'
    ws = GeneralizedSVDWs(A2, B2)

    U1, V1, Q1, alpha1, beta1, k1, l1, R1 = LAPACK.ggsvd3!('U', 'V', 'Q', A1, B1)
    U2, V2, Q2, alpha2, beta2, k2, l2, R2 = LAPACK.ggsvd3!(ws, 'U', 'V', 'Q', A2, B2)

    @test isapprox(U1, U2)
    @test isapprox(V1, V2)
    @test isapprox(Q1, Q2)
    @test isapprox(alpha1, alpha2)
    @test isapprox(beta1, beta2)
    @test k1 == k2
    @test l1 == l2
    @test isapprox(R1, R2)

    # all possible jobs
    for j1 in ('U', 'N'), j2 in ('V', 'N'), j3 in ('Q', 'N')
        A1 = randn(numbertype, m, n)
        A2 = copy(A1)
        B1 = randn(numbertype, m, n)
        B2 = copy(B1)

        ws = GeneralizedSVDWs(A2, B2, jobu = j1, jobv = j2, jobq = j3)

        U1, V1, Q1, alpha1, beta1, k1, l1, R1 = LAPACK.ggsvd3!(j1, j2, j3, A1, B1)
        U2, V2, Q2, alpha2, beta2, k2, l2, R2 = LAPACK.ggsvd3!(ws, j1, j2, j3, A2, B2)

        j1 == 'U' && (@test isapprox(U1, U2))
        j2 == 'V' && (@test isapprox(V1, V2))
        j3 == 'Q' && (@test isapprox(Q1, Q2))
        @test isapprox(alpha1, alpha2)
        @test isapprox(beta1, beta2)
        @test k1 == k2
        @test l1 == l2
        @test isapprox(R1, R2)
    end
end

@testset "SVDsddWs" begin
    @testset "Real, square" test_sdd(10, 10, Float64)
    @testset "Real, nrows < ncols" test_sdd(5, 10, Float64)
    @testset "Real, nrows > ncols" test_sdd(10, 5, Float64)
    @testset "Complex, square" test_sdd(10, 10, ComplexF64)
    @testset "Complex, nrows < ncols" test_sdd(5, 10, ComplexF64)
    @testset "Complex, nrows > ncols" test_sdd(10, 5, ComplexF64)
end
@testset "SVDsvdWs" begin
    @testset "Real, square" test_svd(10, 10, Float64)
    @testset "Real, nrows < ncols" test_svd(5, 10, Float64)
    @testset "Real, nrows > ncols" test_svd(10, 5, Float64)
    @testset "Complex, square" test_svd(10, 10, ComplexF64)
    @testset "Complex, nrows < ncols" test_svd(5, 10, ComplexF64)
    @testset "Complex, nrows > ncols" test_svd(10, 5, ComplexF64)
end
@testset "GeneralizedSVDWs" begin
    @testset "Real, square" test_generalizedsvd(10, 10, Float64)
    @testset "Real, nrows < ncols" test_generalizedsvd(5, 10, Float64)
    @testset "Real, nrows > ncols" test_generalizedsvd(10, 5, Float64)
    @testset "Complex, square" test_generalizedsvd(10, 10, ComplexF64)
    @testset "Complex, nrows < ncols" test_generalizedsvd(5, 10, ComplexF64)
    @testset "Complex, nrows > ncols" test_generalizedsvd(10, 5, ComplexF64)
end
