# [LAPACK](@id LAPACK)

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
