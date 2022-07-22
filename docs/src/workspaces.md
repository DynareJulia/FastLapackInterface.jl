# [Workspaces](@id WorkSpaces)
`Workspaces` represent the buffers and temporary storage that are used during the computations of [`LAPACK functions`](@ref LAPACK).
Upon initialization with a template matrix, work buffers will be allocated that are appropriate to be used during the factorization of matrices similar to the template, e.g. both `Float64` and `Float32` Matrices work, but also `Complex` numbers are allowed when appropriate.

## Workspace
The following convenience function is supplied in order to construct the correct `Workspace` for a given [`LAPACK function`](@ref LAPACK).
This can then be used to perform the decompositions without extra allocations.
```@docs
Workspace
```

## [QR](@id QR-id)

```@docs
QRWs
QRWYWs
QRpWs
```

## [Schur](@id Schur-id)

```@docs
SchurWs
GeneralizedSchurWs
```

## [LU](@id LU-id)
```@docs
LUWs
```

## [Eigen](@id Eigen-id)
```@docs
EigenWs
HermitianEigenWs
GeneralizedEigenWs
```

## [BunchKaufman](@id BunchKaufman-id)
```@docs
BunchKaufmanWs
```

## [Cholesky](@id Cholesky-id)
```@docs
CholeskyPivotedWs
```
