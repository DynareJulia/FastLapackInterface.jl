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
            qr1 = QR(LAPACK.geqrf!(ws, A)...)

            qr2 = QR(LAPACK.geqrf!(copy(A0), randn(n))...)
            @test isapprox(Matrix(qr1), Matrix(qr2))
            # using Workspace, factorize!
            ws = Workspace(LAPACK.geqrf!, copy(A0))
            qr2 = QR(factorize!(ws, copy(A0))...)
            
            @test isapprox(Matrix(qr1), Matrix(qr2))
            
            @test_throws ArgumentError factorize!(ws, rand(n+1, n+1); resize=false)
            factorize!(ws, rand(n+1, n+1))
            @test size(ws.τ , 1) == n+1
        end

        @testset "ormqr!" begin
            ws = Workspace(LAPACK.geqrf!, copy(A0))
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
        @testset "orgqr!" begin
            ws = Workspace(LAPACK.geqrf!, copy(A0))
            C = randn(n, n)
            tau = randn(n)
            ws.τ .= tau
            Cout = LAPACK.orgqr!(copy(A0), tau)
            Cout2 = LAPACK.orgqr!(ws, copy(A0))
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

            qr1 = LinearAlgebra.QRCompactWY(LAPACK.geqrt!(ws, copy(A))...)
            qr2 = LinearAlgebra.QRCompactWY(LAPACK.geqrt!(copy(A0), zeros(size(ws.T)))...)
            @test isapprox(Matrix(qr1), Matrix(qr2))
            # using Workspace, factorize!
            ws = Workspace(LAPACK.geqrt!, copy(A))
            qr1 = LinearAlgebra.QRCompactWY(factorize!(ws, A)...)
            @test isapprox(Matrix(qr1), Matrix(qr2))
            show(devnull, "text/plain", ws)
            @test_throws ArgumentError factorize!(ws, rand(n-1, n-1); resize=false)
            factorize!(ws, rand(n-1, n-1))
            @test size(ws.T, 1) == n-1
        end
    end
end

@testset "QRPivotedWs" begin
    n = 3
    @testset "Real, square" begin
        A0 = randn(n, n)

        @testset "geqp3!" begin
            A = copy(A0)
            ws = QRPivotedWs(A)
            q1 = QRPivoted(LAPACK.geqp3!(ws, copy(A))...)

            q2 = QRPivoted(LAPACK.geqp3!(copy(A0), zeros(Int, length(ws.jpvt)),
                                              zeros(size(ws.τ)))...)
            @test isapprox(Matrix(q1), Matrix(q2))

            # using Workspace, factorize!
            ws = Workspace(LAPACK.geqp3!, copy(A))
            q1 = QRPivoted(factorize!(ws, copy(A))...)
            @test isapprox(Matrix(q1), Matrix(q2))

            @test_throws ArgumentError factorize!(ws, rand(n+1, n+1); resize=false)
            factorize!(ws, rand(n+1, n+1))
            @test size(ws.τ , 1) == n+1
            show(devnull, "text/plain", ws)
        end
        @testset "orgqr!" begin
            ws = Workspace(LAPACK.geqrf!, copy(A0))
            C = randn(n, n)
            tau = randn(n)
            ws.τ .= tau
            Cout = LAPACK.orgqr!(copy(A0), tau)
            Cout2 = LAPACK.orgqr!(ws, copy(A0))
            @test isapprox(Cout, Cout2)
        end
    end
end
