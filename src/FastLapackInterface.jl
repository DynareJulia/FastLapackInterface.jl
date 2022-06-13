module FastLapackInterface

import Base.strides

using LinearAlgebra
using LinearAlgebra: BlasInt, BlasFloat, checksquare, chkstride1
using LinearAlgebra.BLAS: @blasfunc
using LinearAlgebra.LAPACK: chklapackerror

@static if VERSION < v"1.7"
    using LinearAlgebra.LAPACK: liblapack
else
    const liblapack = "libblastrampoline"
end

include("lu.jl")
export LinSolveWs, linsolve_core!, linsolve_core_no_lu!, lu!
include("qr.jl")
export QrWs, QrpWs, geqrf_core!, geqp3!, ormqr_core!
include("schur.jl")
export DgeesWs, dgees!, DggesWs, dgges!

end #module
#import LinearAlgebra: USE_BLAS64, LAPACKException
