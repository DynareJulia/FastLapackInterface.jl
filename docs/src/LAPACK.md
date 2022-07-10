# [LAPACK](@id LAPACK)

## Unified Interface
After having created the [`Workspace`](@ref WorkSpaces) that corresponds to the targeted factorization or decomposition,
one of the following two _aliases_ can be used to dispatch the call to the correct LAPACK function.

```@docs
decompose!
factorize!
```

## QR
```@docs
LinearAlgebra.LAPACK.geqrf!(::QRWs, ::AbstractMatrix)
LinearAlgebra.LAPACK.ormqr!(::QRWs, ::AbstractChar, ::AbstractChar, ::AbstractMatrix, ::AbstractVecOrMat)
LinearAlgebra.LAPACK.geqrt!(::QRWYWs, ::AbstractMatrix)
LinearAlgebra.LAPACK.geqp3!(::QRpWs, ::AbstractMatrix)
```

## Schur
```@docs
LinearAlgebra.LAPACK.gees!(::SchurWs, ::AbstractChar, ::AbstractMatrix)
LinearAlgebra.LAPACK.gges!(::GeneralizedSchurWs, ::AbstractChar, ::AbstractChar, ::AbstractMatrix, ::AbstractMatrix)
```

## LU
```@docs
LinearAlgebra.LAPACK.getrf!(::LUWs, ::AbstractMatrix)
```

## Eigen
```@docs
LinearAlgebra.LAPACK.geevx!(::EigenWs, ::AbstractChar, ::AbstractChar, ::AbstractChar, ::AbstractChar, ::AbstractMatrix)
LinearAlgebra.LAPACK.syevr!(::HermitianEigenWs, ::AbstractChar, ::AbstractChar,
       ::AbstractChar, ::AbstractMatrix, ::AbstractFloat, ::AbstractFloat, ::Integer, ::Integer,
       ::AbstractFloat)
LinearAlgebra.LAPACK.ggev!(::GeneralizedEigenWs, ::AbstractChar, ::AbstractChar, ::AbstractMatrix,
      ::AbstractMatrix)
```
