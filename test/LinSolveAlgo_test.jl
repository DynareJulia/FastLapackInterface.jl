n = 3
m = 2

for elty in (Float32, Float64, ComplexF32, ComplexF64)

    A0 = randn(elty, n, n)
    B0 = randn(elty, n, m)
    B1 = randn(elty, n, m)

    A = copy(A0)

    # Full matrix
    linws = FastLapackInterface.LinSolveWs{elty,Int64}(n)

    FastLapackInterface.lu!(A, linws)
    @test A == A0
    F = lu(A)
    @test UpperTriangular(reshape(linws.lu, n, n)) ≈ F.U

    FastLapackInterface.lu!(A', linws)
    @test UpperTriangular(reshape(linws.lu, n, n)) ≈ F.U

    B = copy(B0)
    FastLapackInterface.linsolve_core!(A, B, linws)
    @test A == A0
    @test B ≈ A \ B0

    copy!(B, B1)
    FastLapackInterface.linsolve_core_no_lu!(A, B, linws)
    @test A == A0
    @test B ≈ A \ B1

    copy!(B, B0)
    FastLapackInterface.linsolve_core!(A', B, linws)
    @test B ≈ A' \ B0

    copy!(B, B1)
    FastLapackInterface.linsolve_core_no_lu!(A', B, linws)
    @test A == A0
    @test B ≈ A' \ B1

    # view of a matrix in upper left corner
    linws1 = FastLapackInterface.LinSolveWs{elty,Int64}(n - 1)
    C = view(A, 1:n-1, 1:n-1)
    D = view(B, 1:n-1, 1:m-1)
    D0 = copy(D)
    D1 = copy(D)
    FastLapackInterface.linsolve_core!(C, D, linws1)
    @test C == view(A, 1:n-1, 1:n-1)
    @test D ≈ C \ D0

    D = copy(D1)
    FastLapackInterface.linsolve_core_no_lu!(C, D, linws1)
    @test C == view(A, 1:n-1, 1:n-1)
    @test D ≈ C \ D1

    D = copy(D0)
    FastLapackInterface.linsolve_core!(C', D, linws1)
    @test C == view(A, 1:n-1, 1:n-1)
    @test D ≈ C' \ D0

    D = copy(D1)
    FastLapackInterface.linsolve_core_no_lu!(C', D, linws1)
    @test C == view(A, 1:n-1, 1:n-1)
    @test D ≈ C' \ D1

    # view of a matrix in lower left corner
    linws1 = FastLapackInterface.LinSolveWs{elty,Int64}(n - 1)
    C = view(A, 2:n, 1:n-1)
    C1 = copy(C)
    FastLapackInterface.lu!(C1, linws1)
    F = LinearAlgebra.lu!(C)
    @test triu(reshape(linws1.lu, n - 1, n - 1)) ≈ F.U
    @test tril(reshape(linws1.lu, n - 1, n - 1), -1) ≈ tril(F.L, -1)

    D0 = view(B0, 2:n, 1:m-1)
    D1 = view(B1, 2:n, 1:m-1)
    D = copy(D0)
    FastLapackInterface.linsolve_core!(C, D, linws1)
    @test C == view(A, 2:n, 1:n-1)
    @test D ≈ C \ D0

    D = copy(D1)
    FastLapackInterface.linsolve_core_no_lu!(C, D, linws1)
    @test C == view(A, 2:n, 1:n-1)
    @test D ≈ C \ D1

    D = copy(D0)
    FastLapackInterface.linsolve_core!(C', D, linws1)
    @test C == view(A, 2:n, 1:n-1)
    @test D ≈ C' \ D0

    D = copy(D1)
    FastLapackInterface.linsolve_core_no_lu!(C', D, linws1)
    @test C == view(A, 2:n, 1:n-1)
    @test D ≈ C' \ D1

    # using too big a work space
    linws1 = FastLapackInterface.LinSolveWs{elty,Int64}(n)
    C = view(A, 2:n, 1:n-1)
    C1 = copy(C)
    FastLapackInterface.lu!(C1, linws1)
    F = LinearAlgebra.lu!(C)
    @test triu(reshape(linws1.lu[1:(n-1)^2], n - 1, n - 1)) ≈ F.U
    @test tril(reshape(linws1.lu[1:(n-1)^2], n - 1, n - 1), -1) ≈ tril(F.L, -1)

    D0 = view(B0, 2:n, 1:m-1)
    D1 = view(B1, 2:n, 1:m-1)
    D = copy(D0)
    FastLapackInterface.linsolve_core!(C, D, linws1)
    @test C == view(A, 2:n, 1:n-1)
    @test D ≈ C \ D0

    D = copy(D1)
    FastLapackInterface.linsolve_core_no_lu!(C, D, linws1)
    @test C == view(A, 2:n, 1:n-1)
    @test D ≈ C \ D1

    D = copy(D0)
    FastLapackInterface.linsolve_core!(C', D, linws1)
    @test C == view(A, 2:n, 1:n-1)
    @test D ≈ C' \ D0

    D = copy(D1)
    FastLapackInterface.linsolve_core_no_lu!(C', D, linws1)
    @test C == view(A, 2:n, 1:n-1)
    @test D ≈ C' \ D1

end
