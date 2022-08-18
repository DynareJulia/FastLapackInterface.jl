using Test
using FastLapackInterface
using LinearAlgebra.LAPACK

@testset "EigenWs" begin
    n = 3
    @testset "Real, square" begin
        A0 = randn(n, n)
        ws = EigenWs(copy(A0); lvecs = true, sense = true)

        A1, WR1, WI1, VL1, VR1, ilo1, ihi1, scale1, abnrm1, rconde1, rcondv1 =
            LAPACK.geevx!('P', 'V', 'V', 'V', copy(A0))
        A2, WR2, WI2, VL2, VR2, ilo2, ihi2, scale2, abnrm2, rconde2, rcondv2 =
            LAPACK.geevx!(ws, 'P', 'V', 'V', 'V', copy(A0))

        @test isapprox(A1, A2)
        @test isapprox(WR1, WR2)
        @test isapprox(WI1, WI2)
        @test isapprox(VL1, VL2)
        @test isapprox(VR1, VR2)
        @test isapprox(ilo1, ilo2)
        @test isapprox(ihi1, ihi2)
        @test isapprox(scale1, scale2)
        @test isapprox(abnrm1, abnrm2)
        @test isapprox(rcondv1, rcondv2; atol = 1e-16)

        # using Workspace, factorize!
        ws = Workspace(LAPACK.geevx!, copy(A0); lvecs = true, sense = true)
        A2, WR2, WI2, VL2, VR2, ilo2, ihi2, scale2, abnrm2, rconde2, rcondv2 =
            factorize!(ws, 'P', 'V', 'V', 'V', copy(A0))
        @test isapprox(A1, A2)
        @test isapprox(WR1, WR2)
        @test isapprox(WI1, WI2)
        @test isapprox(VL1, VL2)
        @test isapprox(VR1, VR2)
        @test isapprox(ilo1, ilo2)
        @test isapprox(ihi1, ihi2)
        @test isapprox(scale1, scale2)
        @test isapprox(abnrm1, abnrm2)
        @test isapprox(rcondv1, rcondv2; atol = 1e-16)
        
        ws = Workspace(LAPACK.geevx!, copy(A0); lvecs=false, rvecs=false, sense=false)
        @test size(ws.VL, 1) == 0
        @test_throws ArgumentError factorize!(ws, 'P', 'V', 'N', 'N', copy(A0); resize=false)
        factorize!(ws, 'P', 'V', 'N', 'N', copy(A0))
        @test size(ws.VL, 1) != 0
        @test_throws ArgumentError factorize!(ws, 'P', 'V', 'V', 'N', copy(A0); resize=false)
        factorize!(ws, 'P', 'V', 'V', 'N', copy(A0))
        @test size(ws.VR, 1) != 0
        @test_throws ArgumentError factorize!(ws, 'P', 'V', 'V', 'E', copy(A0); resize=false)
        factorize!(ws, 'P', 'V', 'V', 'E', copy(A0))
        @test size(ws.iwork, 1) != 0
        @test_throws ArgumentError factorize!(ws, 'P', 'N', 'N', 'N', rand(n+1, n+1); resize=false)
        A2, WR2, WI2, VL2, VR2, ilo2, ihi2, scale2, abnrm2, rconde2, rcondv2 =
            factorize!(ws, 'P', 'N', 'N', 'N', rand(n+1, n+1))

        @test length(WR2) == n+1
       
        show(devnull, "text/plain", ws)
    end

    @testset "Complex, square" begin
        A0 = randn(ComplexF64, n, n)
        ws = EigenWs(copy(A0); lvecs = true, sense = true)

        A1, W1, VL1, VR1, ilo1, ihi1, scale1, abnrm1, rconde1, rcondv1 = LAPACK.geevx!('N',
                                                                                       'V',
                                                                                       'V',
                                                                                       'V',
                                                                                       copy(A0))
        A2, W2, VL2, VR2, ilo2, ihi2, scale2, abnrm2, rconde2, rcondv2 = LAPACK.geevx!(ws,
                                                                                       'N',
                                                                                       'V',
                                                                                       'V',
                                                                                       'V',
                                                                                       copy(A0))

        @test isapprox(A1, A2)
        @test isapprox(W1, W2)
        @test isapprox(VL1, VL2)
        @test isapprox(VR1, VR2)
        @test isapprox(ilo1, ilo2)
        @test isapprox(ihi1, ihi2)
        @test isapprox(scale1, scale2)
        @test isapprox(abnrm1, abnrm2)
        # using Workspace, factorize!
        ws = Workspace(LAPACK.geevx!, copy(A0); lvecs = true, sense = true)
        A2, W2, VL2, VR2, ilo2, ihi2, scale2, abnrm2, rconde2, rcondv2 = factorize!(ws, 'N',
                                                                                    'V',
                                                                                    'V',
                                                                                    'V',
                                                                                    copy(A0))
        @test isapprox(A1, A2)
        @test isapprox(W1, W2)
        @test isapprox(VL1, VL2)
        @test isapprox(VR1, VR2)
        @test isapprox(ilo1, ilo2)
        @test isapprox(ihi1, ihi2)
        @test isapprox(scale1, scale2)
        @test isapprox(abnrm1, abnrm2)
        show(devnull, "text/plain", ws)
    end
end

@testset "HermitianEigenWs" begin
    n = 3
    @testset "Real, square" begin
        A0 = randn(n, n)
        A0 = (A0 + A0')/2
        ws = HermitianEigenWs(copy(A0); vecs = true)

        w1, Z1 = LAPACK.syevr!('V', 'A', 'U', copy(A0), 0.0, 0.0, 0, 0, 1e-6)
        w2, Z2 = LAPACK.syevr!(ws, 'V', 'A', 'U', copy(A0), 0.0, 0.0, 0, 0, 1e-6)
        @test isapprox(w1, w2)
        @test isapprox(Z2, Z2)
        # using Workspace, factorize!
        ws = Workspace(LAPACK.syevr!, copy(A0); vecs = true)
        w2, Z2 = factorize!(ws, 'V', 'A', 'U', copy(A0), 0.0, 0.0, 0, 0, 1e-6)
        @test isapprox(w1, w2)
        @test isapprox(Z2, Z2)
        show(devnull, "text/plain", ws)
        
        ws = Workspace(LAPACK.syevr!, copy(A0); vecs = false)
        @test_throws ArgumentError factorize!(ws, 'N', 'A', 'U', randn(n+1, n+1), 0.0, 0.0, 0, 0, 1e-6; resize=false)
        @test_throws ArgumentError factorize!(ws, 'V', 'A', 'U', copy(A0), 0.0, 0.0, 0, 0, 1e-6; resize=false)
        w2, Z2 = factorize!(ws, 'V', 'A', 'U', randn(n+1, n+1), 0.0, 0.0, 0, 0, 1e-6)
        @test length(ws.w) == n+1
        @test size(ws.Z, 1) == n+1
        w2, Z2 = factorize!(ws, 'V', 'A', 'U', copy(A0), 0.0, 0.0, 0, 0, 1e-16)
        @test length(w2) == n
        @test sum(abs.(Matrix(Eigen(w2, Z2)) .- A0)) < 1e-12
        
        
        @test_throws ArgumentError factorize!(ws, 'V', 'I', 'U', randn(n+1, n+1), 0.0, 0.0, 10, 5, 1e-6)
        @test_throws ArgumentError factorize!(ws, 'V', 'V', 'U', randn(n+1, n+1), 2.0, 1.0, 0, 0, 1e-6)
    end

    @testset "Complex, square" begin
        A0 = randn(ComplexF64, n, n)
        A0 = (A0 + A0') / 2
        ws = HermitianEigenWs(copy(A0); vecs = true)

        w1, Z1 = LAPACK.syevr!('V', 'A', 'U', copy(A0), 0.0, 0.0, 0, 0, 1e-6)
        w2, Z2 = LAPACK.syevr!(ws, 'V', 'A', 'U', copy(A0), 0.0, 0.0, 0, 0, 1e-6)
        @test isapprox(w1, w2)
        @test isapprox(Z2, Z2)
        # using Workspace, factorize!
        ws = Workspace(LAPACK.syevr!, copy(A0); vecs = true)
        w2, Z2 = factorize!(ws, 'V', 'A', 'U', copy(A0), 0.0, 0.0, 0, 0, 1e-6)
        @test isapprox(w1, w2)
        @test isapprox(Z2, Z2)
    end
end
@testset "GeneralizedEigenWs" begin
    n = 3
    @testset "Real, square" begin
        A0 = randn(n, n)
        B0 = randn(n, n)
        ws = GeneralizedEigenWs(copy(A0); lvecs = true, rvecs = true)

        αr1, αi1, β1, vl1, vr1 = LAPACK.ggev!('V', 'V', copy(A0), copy(B0))
        αr2, αi2, β2, vl2, vr2 = LAPACK.ggev!(ws, 'V', 'V', copy(A0), copy(B0))
        @test isapprox(αr1, αr2)
        @test isapprox(αi1, αi2)
        @test isapprox(β1, β2)
        @test isapprox(vl1, vl2)
        @test isapprox(vr1, vr2)
        # using Workspace, factorize!
        ws = Workspace(LAPACK.ggev!, copy(A0); lvecs = true, rvecs = true)
        αr2, αi2, β2, vl2, vr2 = LAPACK.ggev!(ws, 'V', 'V', copy(A0), copy(B0))
        @test isapprox(αr1, αr2)
        @test isapprox(αi1, αi2)
        @test isapprox(β1, β2)
        @test isapprox(vl1, vl2)
        @test isapprox(vr1, vr2)
        show(devnull, "text/plain", ws)
        
        ws = Workspace(LAPACK.ggev!, copy(A0))
        @test size(ws.vl, 1) == 0
        @test size(ws.vr, 1) == 0
        @test_throws ArgumentError LAPACK.ggev!(ws, 'N', 'V', copy(A0), copy(B0); resize=false)
        LAPACK.ggev!(ws, 'N', 'V', copy(A0), copy(B0))
        @test size(ws.vr, 1) == n
        @test_throws ArgumentError LAPACK.ggev!(ws, 'V', 'V', copy(A0), copy(B0); resize=false)
        LAPACK.ggev!(ws, 'V', 'V', copy(A0), copy(B0))
        @test size(ws.vl, 1) == n
        @test_throws ArgumentError LAPACK.ggev!(ws, 'V', 'V', randn(n+1,n+1), randn(n+1, n+1), resize=false)
        LAPACK.ggev!(ws, 'V', 'V', randn(n+1,n+1), randn(n+1, n+1))
        @test size(ws.vl, 1) == n+1
        @test size(ws.vr, 1) == n+1
        αr1, αi1, β1, vl1, vr1 = LAPACK.ggev!(ws, 'V', 'V', copy(A0), copy(B0))
        @test length(αr1) == n
    end

    @testset "Complex, square" begin
        A0 = randn(ComplexF64, n, n)
        B0 = randn(ComplexF64, n, n)
        ws = GeneralizedEigenWs(copy(A0); lvecs = true, rvecs = true)

        αr1, β1, vl1, vr1 = LAPACK.ggev!('V', 'V', copy(A0), copy(B0))
        αr2, β2, vl2, vr2 = LAPACK.ggev!(ws, 'V', 'V', copy(A0), copy(B0))
        @test isapprox(αr1, αr2)
        @test isapprox(β1, β2)
        @test isapprox(vl1, vl2)
        @test isapprox(vr1, vr2)
        # using Workspace, factorize!
        ws = Workspace(LAPACK.ggev!, copy(A0); lvecs = true, rvecs = true)
        αr2, β2, vl2, vr2 = LAPACK.ggev!(ws, 'V', 'V', copy(A0), copy(B0))
        @test isapprox(αr1, αr2)
        @test isapprox(β1, β2)
        @test isapprox(vl1, vl2)
        @test isapprox(vr1, vr2)
        show(devnull, "text/plain", ws)
    end
end
