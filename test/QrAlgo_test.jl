using Test
using FastLapackInterface
using LinearAlgebra.LAPACK

n = 3
#for elty in (Float32, Float64, ComplexF32, ComplexF64)

#    A0 = randn(elty, n, n)

# @static if VERSION < v"1.7"
#     target = qr(A0, Val(true))
# else
#     target = qr(A0, ColumnNorm())
# end

using LinearAlgebra
@testset "QRWs" begin
    A0 = randn(n, n)
    tau1 = zeros(n)
    
    A = copy(A0)
    ws = FastLapackInterface.QRWs(A)
    AT, tau = FastLapackInterface.geqrf!(A, ws)

    AT1, tau1 = LAPACK.geqrf!(copy(A0), tau1)
    @test AT1 == AT
    @test tau1 == tau

    C = randn(n, n)
    Cout = LAPACK.ormqr!('L', 'N', copy(A0), tau1, copy(C))
    Cout2 = FastLapackInterface.ormqr!('L', 'N', copy(A0), copy(C), ws)
    @test isapprox(Cout, Cout2)

    # Is more testing required?
    Cout2 = FastLapackInterface.ormqr!('L', 'T', copy(A0)', copy(C), ws)
    @test isapprox(Cout, Cout2)

end

@testset "QRWsWY" begin
    A0 = randn(n, n)

    A = copy(A0)
    t = FastLapackInterface.QRWsWY(A)
    AT, taut = FastLapackInterface.geqrt!(A, t)

    AT1, taut1 = LinearAlgebra.LAPACK.geqrt!(copy(A0), zeros(size(t.T)))
    @test AT1 == AT
    @test isapprox(taut1, taut)
end

# tau = copy(t.τ)

# @show A0
# @show tau

# @show t.factors
# @show t.τ 
# @show target.factors

# ormqr_core!('L', vA', A, ws)
