using Test
using FastLapackInterface

n = 3
#for elty in (Float32, Float64, ComplexF32, ComplexF64)

#    A0 = randn(elty, n, n)

# @static if VERSION < v"1.7"
#     target = qr(A0, Val(true))
# else
#     target = qr(A0, ColumnNorm())
# end


using LinearAlgebra
@testset "basic" begin 
    A0 = randn(n, n)
    A = copy(A0)

    t = FastLapackInterface.QRWs(A)
    AT, taut = FastLapackInterface.geqrf!(A, t)
    
    AT1, taut1 = LinearAlgebra.LAPACK.geqrf!(A0, zeros(length(taut)))
    @test AT1 == AT
    @test taut1 == taut
end

@testset "new_QR" begin 
    A0 = randn(n, n)
    A = copy(A0)

    t = FastLapackInterface.QRWsNew(A)
    AT, taut = FastLapackInterface.geqrt!(A, t)
    
    AT1, taut1 = LinearAlgebra.LAPACK.geqrt!(A0, zeros(size(t.T)))
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
