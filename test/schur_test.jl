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
    B = real.(FA.vectors * diagm(FA.values) * inv(FA.vectors))
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
        for div in (-1, 1)
            @test_throws FastLapackInterface.WorkspaceSizeError factorize!(
                ws, 'V', randn(n + div, n + div); resize = false)
            factorize!(ws, 'V', randn(n + div, n + div))
            @test length(ws.wr) == n + div
        end
    end

    if Sys.ARCH != :aarch64
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
    end
    @testset "Select keywords" begin
        n = 10
        A0, nunstable = make_matrix(n)
        ws = SchurWs(copy(A0))
        LAPACK.gees!(ws, 'V', copy(A0), select = FastLapackInterface.lp, criterium = -1e-6)
        @test count(real(ws.eigen_values) .< -1e-6) == ws.sdim[]
        LAPACK.gees!(ws, 'V', copy(A0), select = FastLapackInterface.rp, criterium = -1e-6)
        @test count(real(ws.eigen_values) .>= -1e-6) == ws.sdim[]
        LAPACK.gees!(
            ws, 'V', copy(A0), select = FastLapackInterface.id, criterium = 1 + 1e-6)
        @test count(abs.(ws.eigen_values) .< 1 + 1e-6) == ws.sdim[]
        LAPACK.gees!(
            ws, 'V', copy(A0), select = FastLapackInterface.ed, criterium = 1 + 1e-6)
        @test count(abs.(ws.eigen_values) .>= 1 + 1e-6) == ws.sdim[]
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

        for div in (-1, 1)
            @test_throws FastLapackInterface.WorkspaceSizeError factorize!(
                ws, 'V', 'V', randn(n + div, n + div),
                randn(n + div, n + div); resize = false)
            factorize!(ws, 'V', 'V', randn(n + div, n + div), randn(n + div, n + div))
            @test length(ws.αr) == n + div
        end
    end

    #TODO: This should be tested with something realistic
    @show Sys.ARCH
    if Sys.ARCH != :aarch64
        @testset "Real, square, select" begin
            A0 = [-1.1189 -1.1333 -0.985796
                  1.32901 0.628691 0.651912
                  0.924541 0.287144 -1.09629]

            B0 = [-0.256655 -0.626563 -0.948712
                  0.00727555 0.701693 -0.498145
                  0.86268 -0.212549 -0.211994]
            ws = GeneralizedSchurWs(copy(A0))

            CRITERIUM = 1 + 1e-6

            A0, B0, α, β, vsl, vsr = LAPACK.gges!(ws, 'V', 'V', copy(A0),
                copy(B0);
                select = (ar, ai, b) -> ar^2 + ai^2 <
                                        CRITERIUM *
                                        b^2)
            A0, B0, α, β, vsl, vsr = LAPACK.gges!(ws, 'V', 'V', copy(A0),
                copy(B0);
                select = FastLapackInterface.id,
                criterium = CRITERIUM)
            @test ws.sdim[] == 1
            @test (real(α[1]) / β[1])^2 < CRITERIUM
            @test (real(α[2]) / β[2])^2 > CRITERIUM
            @test (real(α[3]) / β[3])^2 > CRITERIUM
            @test isapprox(sort(abs.(eigen(A0, B0).values)), sort(abs.(ws.eigen_values)))
            show(devnull, "text/plain", ws)
        end
    end

    @testset "Select keywords" begin
        n = 10
        A0, nunstable = make_matrix(n)
        ws = GeneralizedSchurWs(copy(A0))

        CRITERIUM = -1e-6
        LAPACK.gges!(ws, 'V', 'V', copy(A0), Matrix(1.0 * I(n));
            select = FastLapackInterface.lp,
            criterium = CRITERIUM)
        @test count(real(ws.eigen_values) .< CRITERIUM) == ws.sdim[]
        LAPACK.gges!(ws, 'V', 'V', copy(A0), Matrix(1.0 * I(n));
            select = FastLapackInterface.rp,
            criterium = CRITERIUM)
        CRITERIUM = 1 - 1e-6
        LAPACK.gges!(ws, 'V', 'V', copy(A0), Matrix(1.0 * I(n));
            select = FastLapackInterface.id,
            criterium = CRITERIUM)
        @test count(abs.(ws.eigen_values) .< CRITERIUM) == ws.sdim[]
        LAPACK.gges!(ws, 'V', 'V', copy(A0), Matrix(1.0 * I(n));
            select = FastLapackInterface.ed,
            criterium = CRITERIUM)
        @test count(abs.(ws.eigen_values) .>= CRITERIUM) == ws.sdim[]
    end
end

@testset "GeneralizedOrdSchurWs" begin
    n = 10
    @testset "Complex, square" begin
        for T in (ComplexF64, ComplexF32)
            A0 = randn(T, n, n)
            B0 = randn(T, n, n)
            F = schur(A0, B0)
            select = LinearAlgebra.BlasInt[isodd(i) for i in 1:n]
            ws = GeneralizedOrdSchurWs(copy(A0))

            S1, T1, α1, β1, Q1, Z1 = LAPACK.tgsen!(select, copy(F.S), copy(F.T), copy(F.Q), copy(F.Z))
            S2, T2, α2, β2, Q2, Z2 = LAPACK.tgsen!(ws, select, copy(F.S), copy(F.T), copy(F.Q), copy(F.Z))
            @test isapprox(S1, S2)
            @test isapprox(T1, T2)
            @test isapprox(α1, α2)
            @test isapprox(β1, β2)
            @test isapprox(Q1, Q2)
            @test isapprox(Z1, Z2)

            # Using Workspace, factorize!
            ws = Workspace(LAPACK.tgsen!, copy(A0))
            S2, T2, α2, β2, Q2, Z2 = factorize!(ws, select, copy(F.S), copy(F.T), copy(F.Q), copy(F.Z))
            @test isapprox(S1, S2)
            @test isapprox(Z1, Z2)
            show(devnull, "text/plain", ws)

            for div in (-1, 1)
                F2 = schur(randn(T, n + div, n + div), randn(T, n + div, n + div))
                select2 = LinearAlgebra.BlasInt[isodd(i) for i in 1:(n + div)]
                @test_throws FastLapackInterface.WorkspaceSizeError factorize!(
                    ws, select2, F2.S, F2.T, F2.Q, F2.Z; resize = false)
                factorize!(ws, select2, F2.S, F2.T, F2.Q, F2.Z)
                @test length(ws.α) == n + div
            end
        end
    end
end

@testset "GeneralizedEigenvectorsWs" begin
    n = 10
    @testset "Complex, square" begin
        for T in (ComplexF64, ComplexF32)
            A0 = randn(T, n, n)
            B0 = randn(T, n, n)
            F = schur(A0, B0)
            λs = F.α ./ F.β
            ws = GeneralizedEigenvectorsWs(copy(A0))
            tol = sqrt(eps(real(T)))

            # back-transformed right and left eigenvectors of (A0, B0)
            VL, VR, m = FastLapackInterface.tgevc!(ws, 'B', 'B', LinearAlgebra.BlasInt[],
                F.S, F.T, copy(F.Q), copy(F.Z))
            @test m == n
            for j in 1:n
                v, u = VR[:, j], VL[:, j]
                @test norm(A0 * v - λs[j] * B0 * v) < tol * norm(A0) * norm(v)
                @test norm(u' * A0 - λs[j] * u' * B0) < tol * norm(A0) * norm(u)
            end

            # selected right eigenvectors of (S, T), written into a view
            select = LinearAlgebra.BlasInt[i <= n ÷ 2 for i in 1:n]
            VRall = FastLapackInterface.tgevc!(ws, 'R', 'A', select, F.S, F.T,
                similar(F.S, 0, n), similar(F.S))[2]
            VRsel = zeros(T, n, n)
            _, _, m = FastLapackInterface.tgevc!(ws, 'R', 'S', select, F.S, F.T,
                similar(F.S, 0, n), view(VRsel, :, 1:(n ÷ 2)))
            @test m == n ÷ 2
            @test isapprox(VRsel[:, 1:(n ÷ 2)], VRall[:, 1:(n ÷ 2)])
            @test iszero(VRsel[:, (n ÷ 2 + 1):n])

            # Using Workspace, factorize!
            ws = Workspace(FastLapackInterface.tgevc!, copy(A0))
            _, VR2, _ = factorize!(ws, 'R', 'A', select, F.S, F.T, similar(F.S, 0, n), similar(F.S))
            @test isapprox(VRall, VR2)
            show(devnull, "text/plain", ws)

            @test_throws ArgumentError FastLapackInterface.tgevc!(ws, 'X', 'A', select,
                F.S, F.T, similar(F.S, 0, n), similar(F.S))
            for div in (-1, 1)
                F2 = schur(randn(T, n + div, n + div), randn(T, n + div, n + div))
                args2 = ('R', 'A', LinearAlgebra.BlasInt[], F2.S, F2.T,
                    similar(F2.S, 0, n + div), similar(F2.S))
                @test_throws FastLapackInterface.WorkspaceSizeError factorize!(ws, args2...;
                    resize = false)
                factorize!(ws, args2...)
                @test length(ws) == n + div
            end
        end
    end
end
