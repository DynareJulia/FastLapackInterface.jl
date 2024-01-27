using LinearAlgebra
using LinearAlgebra.LAPACK
n = 10
m = 8

@testset "LU" begin
    for elty in (Float32, Float64, ComplexF32, ComplexF64)
        @testset "$elty" begin
            A0 = randn(elty, n, n)
            B0 = randn(elty, n, m)
            B1 = randn(elty, n, m)

            A = copy(A0)

            # Full matrix
            linws = LUWs(n)

            res = LU(LAPACK.getrf!(linws, copy(A))...)
            F = lu(A0)
            @test F.U ≈ res.U
            @test UpperTriangular(reshape(res.U, n, n)) ≈ F.U
            # Using Workspace, factorize!
            linws = Workspace(LAPACK.getrf!, copy(A))

            res = LU(factorize!(linws, copy(A))...)
            F = lu(A0)
            @test F.U ≈ res.U
            @test UpperTriangular(reshape(res.U, n, n)) ≈ F.U

            @test LAPACK.getrs!('N', res.factors, linws.ipiv, copy(B0)) == LAPACK.getrs!(linws, 'N', res.factors, copy(B0))

            show(devnull, "text/plain", linws)
            for div in (-1, 1)
                @test_throws FastLapackInterface.WorkspaceSizeError factorize!(linws, rand(elty, n+div, n+div); resize=false)
                factorize!(linws, rand(elty, n+div, n+div))
                @test length(linws.ipiv) == n+div
            end

            # res = LU(LAPACK.getrf!(collect(copy(A)'), linws)...)
            # @test UpperTriangular(reshape(res.U, n, n)) ≈ F.U

            # B = copy(B0)
            # FastLapackInterface.linsolve_core!(A, B, linws)
            # @test A == A0
            # @test B ≈ A \ B0

            # copy!(B, B1)
            # FastLapackInterface.linsolve_core_no_lu!(A, B, linws)
            # @test A == A0
            # @test B ≈ A \ B1

            # copy!(B, B0)
            # FastLapackInterface.linsolve_core!(A', B, linws)
            # @test B ≈ A' \ B0

            # copy!(B, B1)
            # FastLapackInterface.linsolve_core_no_lu!(A', B, linws)
            # @test A == A0
            # @test B ≈ A' \ B1

            # # view of a matrix in upper left corner
            # linws1 = FastLapackInterface.LUWs{elty,Int64}(n - 1)
            # C = view(A, 1:n-1, 1:n-1)
            # D = view(B, 1:n-1, 1:m-1)
            # D0 = copy(D)
            # D1 = copy(D)
            # FastLapackInterface.linsolve_core!(C, D, linws1)
            # @test C == view(A, 1:n-1, 1:n-1)
            # @test D ≈ C \ D0

            # D = copy(D1)
            # FastLapackInterface.linsolve_core_no_lu!(C, D, linws1)
            # @test C == view(A, 1:n-1, 1:n-1)
            # @test D ≈ C \ D1

            # D = copy(D0)
            # FastLapackInterface.linsolve_core!(C', D, linws1)
            # @test C == view(A, 1:n-1, 1:n-1)
            # @test D ≈ C' \ D0

            # D = copy(D1)
            # FastLapackInterface.linsolve_core_no_lu!(C', D, linws1)
            # @test C == view(A, 1:n-1, 1:n-1)
            # @test D ≈ C' \ D1

            # # view of a matrix in lower left corner
            # linws1 = FastLapackInterface.LUWs{elty,Int64}(n - 1)
            # C = view(A, 2:n, 1:n-1)
            # C1 = copy(C)
            # FastLapackInterface.lu!(C1, linws1)
            # F = LinearAlgebra.lu!(C)
            # @test triu(reshape(linws1.lu, n - 1, n - 1)) ≈ F.U
            # @test tril(reshape(linws1.lu, n - 1, n - 1), -1) ≈ tril(F.L, -1)

            # D0 = view(B0, 2:n, 1:m-1)
            # D1 = view(B1, 2:n, 1:m-1)
            # D = copy(D0)
            # FastLapackInterface.linsolve_core!(C, D, linws1)
            # @test C == view(A, 2:n, 1:n-1)
            # @test D ≈ C \ D0

            # D = copy(D1)
            # FastLapackInterface.linsolve_core_no_lu!(C, D, linws1)
            # @test C == view(A, 2:n, 1:n-1)
            # @test D ≈ C \ D1

            # D = copy(D0)
            # FastLapackInterface.linsolve_core!(C', D, linws1)
            # @test C == view(A, 2:n, 1:n-1)
            # @test D ≈ C' \ D0

            # D = copy(D1)
            # FastLapackInterface.linsolve_core_no_lu!(C', D, linws1)
            # @test C == view(A, 2:n, 1:n-1)
            # @test D ≈ C' \ D1

            # # using too big a work space
            # linws1 = FastLapackInterface.LUWs{elty,Int64}(n)
            # C = view(A, 2:n, 1:n-1)
            # C1 = copy(C)
            # FastLapackInterface.lu!(C1, linws1)
            # F = LinearAlgebra.lu!(C)
            # @test triu(reshape(linws1.lu[1:(n-1)^2], n - 1, n - 1)) ≈ F.U
            # @test tril(reshape(linws1.lu[1:(n-1)^2], n - 1, n - 1), -1) ≈ tril(F.L, -1)

            # D0 = view(B0, 2:n, 1:m-1)
            # D1 = view(B1, 2:n, 1:m-1)
            # D = copy(D0)
            # FastLapackInterface.linsolve_core!(C, D, linws1)
            # @test C == view(A, 2:n, 1:n-1)
            # @test D ≈ C \ D0

            # D = copy(D1)
            # FastLapackInterface.linsolve_core_no_lu!(C, D, linws1)
            # @test C == view(A, 2:n, 1:n-1)
            # @test D ≈ C \ D1

            # D = copy(D0)
            # FastLapackInterface.linsolve_core!(C', D, linws1)
            # @test C == view(A, 2:n, 1:n-1)
            # @test D ≈ C' \ D0

            # D = copy(D1)
            # FastLapackInterface.linsolve_core_no_lu!(C', D, linws1)
            # @test C == view(A, 2:n, 1:n-1)
            # @test D ≈ C' \ D1
        end
    end
end
