# [LAPACK](@id LAPACK)

## QR
```@docs
LAPACK.geqrf!(::AbstractMatrix, ::QRWs)
LAPACK.ormqr!(::AbstractChar, ::AbstractChar, ::AbstractMatrix, ::AbstractVecOrMat, ::QRWs)
LAPACK.geqrt!(::AbstractMatrix, ::QRWYWs)
LAPACK.geqp3!(::AbstractMatrix, ::QRpWs)
```

## Schur
```@docs
LAPACK.gees!(::AbstractChar, ::AbstractMatrix, ::SchurWs)
LAPACK.gges!(::AbstractChar, ::AbstractChar, ::AbstractMatrix, ::AbstractMatrix, ::GeneralizedSchurWs)
```

## LU
```@docs
LAPACK.getrf!(::AbstractMatrix, ::LUWs)
```
