# [Workspaces](@id WorkSpaces)
`Workspaces` represent the buffers and temporary storage that are used during the computations of [`LAPACK functions`](@ref LAPACK).
Upon initialization with a template matrix, work buffers will be allocated that are appropriate to be used during the factorization of matrices similar to the template, e.g. both `Float64` and `Float32` Matrices work.
## [QR](@id QR-id)

```@docs
QRWs
```

```@docs
QRWYWs
```

```@docs
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
