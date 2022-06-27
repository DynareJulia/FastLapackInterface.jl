# [LAPACK](@id LAPACK)

## QR
```@docs
LinearAlgebra.LAPACK.geqrf!(::AbstractMatrix, ::QRWs)
LinearAlgebra.LAPACK.ormqr!(::AbstractChar, ::AbstractChar, ::AbstractMatrix, ::AbstractVecOrMat, ::QRWs)
LinearAlgebra.LAPACK.geqrt!(::AbstractMatrix, ::QRWYWs)
LinearAlgebra.LAPACK.geqp3!(::AbstractMatrix, ::QRpWs)
```

## Schur
```@docs
LinearAlgebra.LAPACK.gees!(::AbstractChar, ::AbstractMatrix, ::SchurWs)
LinearAlgebra.LAPACK.gges!(::AbstractChar, ::AbstractChar, ::AbstractMatrix, ::AbstractMatrix, ::GeneralizedSchurWs)
```

## LU
```@docs
LinearAlgebra.LAPACK.getrf!(::AbstractMatrix, ::LUWs)
```
