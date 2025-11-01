using FastLapackInterface
using LinearAlgebra
using Test

function test_FastLapackInterface()
    include("lse_test.jl")
    include("lu_test.jl")
    include("schur_test.jl")
    include("qr_test.jl")
    include("eigen_test.jl")
    include("bunch_kaufman_test.jl")
    include("cholesky_test.jl")
    include("svd_test.jl")
end

test_FastLapackInterface()
if Sys.islinux() || Sys.iswindows()
    using MKL
    test_FastLapackInterface()
elseif Sys.isapple()
    using AppleAccelerate
    test_FastLapackInterface()
end
