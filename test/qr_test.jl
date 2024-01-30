using Test
using FastLapackInterface
using LinearAlgebra.LAPACK

@testset "QRWs" begin
    n = 6
    m = 12
    for T in (Float32, Float64, ComplexF32, ComplexF64)
        @testset "$T" begin
            A0 = randn(T, n, n)
            ws = QRWs(A0)
            @testset "geqrf!" begin
                A = copy(A0)
                qr1 = QR(LAPACK.geqrf!(ws, A)...)

                qr2 = QR(LAPACK.geqrf!(copy(A0), randn(T, n))...)
                @test isapprox(Matrix(qr1), Matrix(qr2))
                # using Workspace, factorize!
                ws = Workspace(LAPACK.geqrf!, copy(A0))
                qr2 = QR(factorize!(ws, copy(A0))...)
                
                @test isapprox(Matrix(qr1), Matrix(qr2))
                for div in (-1, 1)            
                    @test_throws FastLapackInterface.WorkspaceSizeError factorize!(ws, rand(T, n+div, n+div); resize=false)
                    factorize!(ws, rand(T, n+div, n+div))
                    @test size(ws.τ , 1) == n+div
                end
            end

            @testset "ormqr!" begin
                AA = copy(A0)
                ws = Workspace(LAPACK.geqrf!, AA)
                C = randn(T, n, m)
                tau = randn(T, n)
                ws.τ .= tau
                ormws = QROrmWs(ws, 'L', 'N', AA, copy(C))
                Cout = LAPACK.ormqr!('L', 'N', AA, tau, copy(C))
                Cout2 = LAPACK.ormqr!(ormws, 'L', 'N', AA, copy(C))
                @test isapprox(Cout, Cout2)

                trans = T <: Complex ? 'C' : 'T'
                ormws = QROrmWs(ws, 'L', trans, AA, copy(C))
                Cout = LAPACK.ormqr!('L', trans, AA, tau, copy(C))
                Cout2 = LAPACK.ormqr!(ormws, 'L', trans, AA, copy(C))
                @test isapprox(Cout, Cout2)

                C = randn(T, m, n)
                ormws = QROrmWs(ws, 'R', 'N', AA, copy(C))
                Cout = LAPACK.ormqr!('R', 'N', AA, tau, copy(C))
                Cout2 = LAPACK.ormqr!(ormws, 'R', 'N', AA, copy(C))
                @test isapprox(Cout, Cout2)

                trans = T <: Complex ? 'C' : 'T'
                ormws = QROrmWs(ws, 'R', trans, AA, copy(C))
                Cout = LAPACK.ormqr!('R', trans, AA, tau, copy(C))
                Cout2 = LAPACK.ormqr!(ormws, 'R', trans, AA, copy(C))
                @test isapprox(Cout, Cout2)
            end
            @testset "orgqr!" begin
                ws = Workspace(LAPACK.geqrf!, copy(A0))
                C = randn(T, n, n)
                tau = randn(T, n)
                ws.τ .= tau
                Cout = LAPACK.orgqr!(copy(A0), tau)
                Cout2 = LAPACK.orgqr!(ws, copy(A0))
                @test isapprox(Cout, Cout2)
            end
            show(devnull, "text/plain", ws)
        end
    end
end

@testset "QRWYWs" begin
    n = 3
    for T in (Float32, Float64, ComplexF32, ComplexF64)
        @testset "$T" begin
            A0 = randn(T, n, n)

            @testset "geqrt!" begin
                A = copy(A0)
                ws = QRWYWs(A)

                qr1 = LinearAlgebra.QRCompactWY(LAPACK.geqrt!(ws, copy(A))...)
                qr2 = LinearAlgebra.QRCompactWY(LAPACK.geqrt!(copy(A0), zeros(T, size(ws.T)))...)
                @test isapprox(Matrix(qr1), Matrix(qr2))
                # using Workspace, factorize!
                ws = Workspace(LAPACK.geqrt!, copy(A))
                qr1 = LinearAlgebra.QRCompactWY(factorize!(ws, A)...)
                @test isapprox(Matrix(qr1), Matrix(qr2))
                show(devnull, "text/plain", ws)
                for div in (-1, 1)            
                    @test_throws FastLapackInterface.WorkspaceSizeError factorize!(ws, rand(T, n+div, n+div); resize=false)
                    factorize!(ws, rand(T, n+div, n+div))
                    @test size(ws.T , 1) == n+div
                end
            end
       end 
    end
end

@testset "QRPivotedWs" begin
    n = 10
    for T in (Float32, Float64, ComplexF32, ComplexF64)
        @testset "$T" begin
            A0 = randn(T, n, n)

            @testset "geqp3!" begin
                A = copy(A0)
                ws = QRPivotedWs(A)
                q1 = QRPivoted(LAPACK.geqp3!(ws, copy(A))...)

                q2 = QRPivoted(LAPACK.geqp3!(copy(A0), zeros(Int, length(ws.jpvt)),
                                                  zeros(T, size(ws.τ)))...)
                @test isapprox(Matrix(q1), Matrix(q2))

                # using Workspace, factorize!
                ws = Workspace(LAPACK.geqp3!, copy(A))
                q1 = QRPivoted(factorize!(ws, copy(A))...)
                @test isapprox(Matrix(q1), Matrix(q2))

                for div in (-1, 1)            
                    @test_throws FastLapackInterface.WorkspaceSizeError factorize!(ws, rand(T, n+div, n+div); resize=false)
                    factorize!(ws, rand(T, n+div, n+div))
                    @test size(ws.τ, 1) == n+div
                end
                show(devnull, "text/plain", ws)
            end
            @testset "orgqr!" begin
                ws = Workspace(LAPACK.geqrf!, copy(A0))
                C = randn(T, n, n)
                tau = randn(T, n)
                ws.τ .= tau
                Cout = LAPACK.orgqr!(copy(A0), tau)
                Cout2 = LAPACK.orgqr!(ws, copy(A0))
                @test isapprox(Cout, Cout2)
            end
        end
    end
end
