module FastLapackInterface

import Base.strides

using Base: require_one_based_indexing
using LinearAlgebra
using LinearAlgebra: BlasInt, BlasFloat, checksquare, chkstride1
using LinearAlgebra.BLAS: @blasfunc
using LinearAlgebra.LAPACK: chklapackerror, chkargsok, chkstride1, chktrans, chkside
using LinearAlgebra.LAPACK

@static if VERSION < v"1.7"
    using LinearAlgebra.LAPACK: liblapack
else
    const liblapack = "libblastrampoline"
end

include("lu.jl")
export LUWs
include("qr.jl")
export QRWs, QRWYWs, QRpWs
include("schur.jl")
export SchurWs, GeneralizedSchurWs
include("eigen.jl")
export EigenWs, HermitianEigenWs

Workspace(::typeof(LAPACK.getrf!), A::AbstractMatrix) = LUWs(A)

Workspace(::typeof(LAPACK.geqrf!), A::AbstractMatrix) = QRWs(A)
Workspace(::typeof(LAPACK.ormqr!), A::AbstractMatrix) = QRWs(A)
Workspace(::typeof(LAPACK.geqrt!), A::AbstractMatrix) = QRWYWs(A)
Workspace(::typeof(LAPACK.geqp3!), A::AbstractMatrix) = QRpWs(A)

Workspace(::typeof(LAPACK.gees!), A::AbstractMatrix) = SchurWs(A)
Workspace(::typeof(LAPACK.gges!), A::AbstractMatrix) = GeneralizedSchurWs(A)

Workspace(::typeof(LAPACK.geevx!), A::AbstractMatrix; kwargs...) = EigenWs(A; kwargs...)
Workspace(::typeof(LAPACK.syevr!), A::AbstractMatrix; kwargs...) = HermitianEigenWs(A; kwargs...)

export Workspace

decompose!(ws::LUWs, args...) = LAPACK.getrf!(ws, args...)

decompose!(ws::QRWs, args...)   = LAPACK.geqrf!(ws, args...)
decompose!(ws::QRWYWs, args...) = LAPACK.geqrt!(ws, args...)
decompose!(ws::QRpWs, args...)  = LAPACK.geqp3!(ws, args...)

decompose!(ws::SchurWs, args...)            = LAPACK.gees!(ws, args...)
decompose!(ws::GeneralizedSchurWs, args...) = LAPACK.gges!(ws, args...)

decompose!(ws::EigenWs, args...)          = LAPACK.geevx!(ws, args...)
decompose!(ws::HermitianEigenWs, args...) = LAPACK.syevr!(ws, args...)

const factorize! = decompose!

export decompose!, factorize!


end #module
#import LinearAlgebra: USE_BLAS64, LAPACKException
