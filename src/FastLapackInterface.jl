module FastLapackInterface

import Base.strides

using Base: require_one_based_indexing
using LinearAlgebra
using LinearAlgebra: BlasInt, BlasFloat, checksquare, chkstride1
using LinearAlgebra.BLAS: @blasfunc
using LinearAlgebra.LAPACK: chklapackerror, chkargsok, chkstride1, chktrans, chkside
using LinearAlgebra.LAPACK

using LinearAlgebra.LAPACK: liblapack

abstract type Workspace end
include("exceptions.jl")
include("lu.jl")
export LUWs
include("qr.jl")
export QRWs, QRWYWs, QRPivotedWs
include("schur.jl")
export SchurWs, GeneralizedSchurWs
include("eigen.jl")
export EigenWs, HermitianEigenWs, GeneralizedEigenWs
include("bunch_kaufman.jl")
export BunchKaufmanWs
include("cholesky.jl")
export CholeskyPivotedWs
include("lse.jl")
export LSEWs

# Uniform interface
include("workspace.jl")
export Workspace
export decompose!, factorize!

end #module
#import LinearAlgebra: USE_BLAS64, LAPACKException
