# [LAPACK](@id LAPACK)
This section details the [`LAPACK
functions`](https://docs.julialang.org/en/v1/stdlib/LinearAlgebra/#LinearAlgebra.LAPACK)
that are supported for use with various [`Workspaces`](@ref
WorkSpaces). Each function has a `resize` keyword argument that is
`true` by default, allowing for automatic resizing of the workspaces
to accomodate larger or smaller Matrices or different features than they were
originally constructed for. This is provided as a convenience but
involves an efficiency cost. When working with matrices of different sizes,
the best strategy is to successively apply a function to all matrices of the same
size and to minimize triggering the resizing mechanism.

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
LinearAlgebra.LAPACK.geqrt!(::QRWYWs, ::AbstractMatrix)
LinearAlgebra.LAPACK.geqp3!(::QRPivotedWs, ::AbstractMatrix)
LinearAlgebra.LAPACK.ormqr!(::QRWs, ::AbstractChar, ::AbstractChar, ::AbstractMatrix, ::AbstractVecOrMat)
LinearAlgebra.LAPACK.orgqr!(::QRWs, ::AbstractMatrix, ::Integer)
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

## BunchKaufman
```@docs
LinearAlgebra.LAPACK.sytrf!(::BunchKaufmanWs, ::AbstractChar, ::AbstractMatrix)
LinearAlgebra.LAPACK.sytrf_rook!(::BunchKaufmanWs, ::AbstractChar, ::AbstractMatrix)
LinearAlgebra.LAPACK.hetrf!(::BunchKaufmanWs, ::AbstractChar, ::AbstractMatrix)
LinearAlgebra.LAPACK.hetrf_rook!(::BunchKaufmanWs, ::AbstractChar, ::AbstractMatrix)
```

## Cholesky
```@docs
LinearAlgebra.LAPACK.pstrf!(::CholeskyPivotedWs, ::AbstractChar, ::AbstractMatrix, ::Real)
```
