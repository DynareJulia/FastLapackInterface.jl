import LinearAlgebra.LAPACK: pstrf!
"""
    CholeskyPivotedWs

Workspace for [`LinearAlgebra.CholeskyPivoted`](https://docs.julialang.org/en/v1/stdlib/LinearAlgebra/#LinearAlgebra.CholeskyPivoted) factorization using the [`LAPACK.pstrf!`](@ref) function.
The standard [`LinearAlgebra.Cholesky`](https://docs.julialang.org/en/v1/stdlib/LinearAlgebra/#LinearAlgebra.Cholesky) uses [`LAPACK.potrf!`](https://docs.julialang.org/en/v1/stdlib/LinearAlgebra/#LinearAlgebra.LAPACK.potrf!) which is non-allocating and does not require a separate [`Workspace`](@ref).

# Examples
```jldoctest
julia> A = [1.2 7.8
            7.8 3.3]
2×2 Matrix{Float64}:
 1.2  7.8
 7.8  3.3

julia> ws = CholeskyPivotedWs(A)
CholeskyPivotedWs{Float64}
  work: 4-element Vector{Float64}
  piv: 2-element Vector{Int64}


julia> AA, piv, rank, info = LAPACK.pstrf!(ws, 'U', A, 1e-6)
([1.816590212458495 4.293758683992806; 7.8 -17.236363636363635], [2, 1], 1, 1)

julia> CholeskyPivoted(AA, 'U', piv, rank, 1e-6, info)
CholeskyPivoted{Float64, Matrix{Float64}, Vector{Int64}}
U factor with rank 1:
2×2 UpperTriangular{Float64, Matrix{Float64}}:
 1.81659    4.29376
  ⋅       -17.2364
permutation:
2-element Vector{Int64}:
 2
 1
``` 
"""
struct CholeskyPivotedWs{T} <: Workspace
    work::Vector{T}
    piv::Vector{BlasInt}
end

for (pstrf, elty, rtyp) in
    ((:dpstrf_,:Float64,:Float64),
     (:spstrf_,:Float32,:Float32),
     (:zpstrf_,:ComplexF64,:Float64),
     (:cpstrf_,:ComplexF32,:Float32))
    @eval begin
        function Base.resize!(ws::CholeskyPivotedWs, A::AbstractMatrix{$elty})
            n = checksquare(A)
            resize!(ws.work, 2n)
            resize!(ws.piv, n)
            return ws
        end
        function CholeskyPivotedWs(A::AbstractMatrix{$elty})
            return resize!(CholeskyPivotedWs($rtyp[], BlasInt[]), A)
        end

        function pstrf!(ws::CholeskyPivotedWs, uplo::AbstractChar, A::AbstractMatrix{$elty}, tol::Real; resize=true)
            chkstride1(A)
            n = checksquare(A)
            chkuplo(uplo)
            rank = Ref{BlasInt}()
            info = Ref{BlasInt}()
            if length(ws.piv) < n
                if resize
                    resize!(ws, A)
                else
                    throw(ArgumentError("Workspace is too small, use resize!(ws, A)."))
                end
            end
                
            ccall((@blasfunc($pstrf), liblapack), Cvoid,
                  (Ref{UInt8}, Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt}, Ptr{BlasInt},
                   Ptr{BlasInt}, Ref{$rtyp}, Ptr{$rtyp}, Ptr{BlasInt}, Clong),
                  uplo, n, A, max(1,stride(A,2)), ws.piv, rank, tol, ws.work, info, 1)
            chkargsok(info[])
            A, ws.piv, rank[], info[]
        end
    end
end

"""
    pstrf!(ws, uplo, A, tol; resize=true) -> (A, ws.piv, rank, info)

Computes the (upper if `uplo = U`, lower if `uplo = L`) pivoted Cholesky
decomposition of positive-definite matrix `A` with a user-set tolerance
`tol`, using a preallocated [`CholeskyPivotedWs`](@ref).
If the workspace was too small and `resize==true` it will be automatically resized.
`A` is overwritten by its Cholesky decomposition.

Returns `A`, the pivots `piv`, the rank of `A`, and an `info` code. If `info = 0`,
the factorization succeeded. If `info = i > 0 `, then `A` is indefinite or
rank-deficient.
"""
pstrf!(ws::CholeskyPivotedWs, uplo::AbstractChar, A::AbstractMatrix, tol::Real; kwargs...)

