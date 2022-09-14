# [Workspaces](@id WorkSpaces)
`Workspaces` represent the buffers and temporary storage that are used during the computations of [`LAPACK functions`](@ref LAPACK).
Upon initialization with a template matrix, work buffers will be allocated that are appropriate to be used during the factorization of matrices similar to the template, e.g. both `Float64` and `Float32` Matrices work, but also `Complex` numbers are allowed when appropriate.

## Workspace
The following convenience function is supplied in order to construct the correct `Workspace` for a given [`LAPACK function`](@ref LAPACK).
This can then be used to perform the decompositions without extra allocations.
```@docs
Workspace
```
Each [`Workspace`](@ref) also has a function to [`resize!`](@ref) to allow for its use with larger matrices or with more features (e.g. the computation of left eigenvectors and right eigenvectors using [`EigenWs`](@ref)).
```@docs
resize!(::Workspace, ::AbstractMatrix; kwargs...)
```

## [QR](@id QR-id)

```@docs
QRWs
QRWYWs
QRPivotedWs
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

## [LSE](@id LSE-id)
```@docs
LSEWs
```
