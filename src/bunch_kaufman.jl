using LinearAlgebra.LAPACK: chkuplo
import LinearAlgebra.LAPACK: sytrf!, sytrf_rook!, hetrf!, hetrf_rook!

"""
    BunchKaufmanWs

Workspace for [`LinearAlgebra.BunchKaufman`](https://docs.julialang.org/en/v1/stdlib/LinearAlgebra/#LinearAlgebra.BunchKaufman)
factorization using the [`LAPACK.sytrf!`](@ref) or [`LAPACK.sytrf_rook!`](@ref) functions for symmetric matrices, and [`LAPACK.hetrf!`](@ref) or [`LAPACK.hetrf_rook!`](@ref) functions for hermitian matrices (e.g. with `ComplexF64` or `ComplexF32` elements).

# Examples
```jldoctest
julia> A = [1.2 7.8
            7.8 3.3]
2×2 Matrix{Float64}:
 1.2  7.8
 7.8  3.3

julia> ws = BunchKaufmanWs(A)
BunchKaufmanWs{Float64}
  work: 128-element Vector{Float64}
  ipiv: 2-element Vector{Int64}


julia> A, ipiv, info = LAPACK.sytrf!(ws, 'U', A)
([1.2 7.8; 7.8 3.3], [-1, -1], 0)

julia> t = LinearAlgebra.BunchKaufman(A, ipiv,'U', true, false, info)
BunchKaufman{Float64, Matrix{Float64}, Vector{Int64}}
D factor:
2×2 Tridiagonal{Float64, Vector{Float64}}:
 1.2  7.8
 7.8  3.3
U factor:
2×2 UnitUpperTriangular{Float64, Matrix{Float64}}:
 1.0  0.0
  ⋅   1.0
permutation:
2-element Vector{Int64}:
 1
 2

julia> A = [1.2 7.8
            7.8 3.3]
2×2 Matrix{Float64}:
 1.2  7.8
 7.8  3.3

julia> ws = BunchKaufmanWs(A)
BunchKaufmanWs{Float64}
  work: 128-element Vector{Float64}
  ipiv: 2-element Vector{Int64}


julia> A, ipiv, info = LAPACK.sytrf_rook!(ws, 'U', A)
([1.2 7.8; 7.8 3.3], [-1, -2], 0)

julia> t = LinearAlgebra.BunchKaufman(A, ipiv,'U', true, true, info)
BunchKaufman{Float64, Matrix{Float64}, Vector{Int64}}
D factor:
2×2 Tridiagonal{Float64, Vector{Float64}}:
 1.2  7.8
 7.8  3.3
U factor:
2×2 UnitUpperTriangular{Float64, Matrix{Float64}}:
 1.0  0.0
  ⋅   1.0
permutation:
2-element Vector{Int64}:
 1
 2
```
"""
struct BunchKaufmanWs{T} <: Workspace
    work::Vector{T}
    ipiv::Vector{BlasInt}
end

for (sytrfs, elty) in (((:dsytrf_, :dsytrf_rook_), :Float64),
    ((:ssytrf_, :ssytrf_rook_), :Float32),
    ((:zsytrf_, :zsytrf_rook_, :zhetrf_, :zhetrf_rook_), :ComplexF64),
    ((:csytrf_, :csytrf_rook_, :chetrf_, :chetrf_rook_), :ComplexF32))
    @eval begin
        function Base.resize!(ws::BunchKaufmanWs, A::AbstractMatrix{$elty}; work = true)
            chkstride1(A)
            n = checksquare(A)
            if n == 0
                return ws
            end
            resize!(ws.ipiv, n)
            if work
                info = Ref{BlasInt}()
                ccall((@blasfunc($(sytrfs[1])), liblapack), Cvoid,
                    (Ref{UInt8}, Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt},
                        Ptr{BlasInt}, Ptr{$elty}, Ref{BlasInt}, Ptr{BlasInt}, Clong),
                    'U', n, A, stride(A, 2), ws.ipiv, ws.work, -1, info, 1)
                chkargsok(info[])
                resize!(ws.work, BlasInt(real(ws.work[1])))
            end
            return ws
        end
        function BunchKaufmanWs(A::AbstractMatrix{$elty})
            return resize!(BunchKaufmanWs(Vector{$elty}(undef, 1), BlasInt[]), A)
        end
    end
    for (sytrf, fn) in zip(sytrfs, (:sytrf!, :sytrf_rook!, :hetrf!, :hetrf_rook!))
        @eval function $fn(ws::BunchKaufmanWs{$elty}, uplo::AbstractChar,
                A::AbstractMatrix{$elty}; resize = true)
            chkstride1(A)
            n = checksquare(A)
            if n == 0
                return A, ws.ipiv, zero(BlasInt)
            end
            chkuplo(uplo)
            nws = length(ws.ipiv)
            if n != nws
                if resize
                    resize!(ws, A, work = nws < n)
                else
                    throw(WorkspaceSizeError(nws, n))
                end
            end
            info = Ref{BlasInt}()
            ccall((@blasfunc($sytrf), liblapack), Cvoid,
                (Ref{UInt8}, Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt},
                    Ptr{BlasInt}, Ptr{$elty}, Ref{BlasInt}, Ptr{BlasInt}, Clong),
                uplo, n, A, stride(A, 2), ws.ipiv, ws.work, length(ws.work), info, 1)
            chkargsok(info[])
            return A, ws.ipiv, info[]
        end
    end
end

"""
    sytrf!(ws, uplo, A; resize=true) -> (A, ws.ipiv, info)

Computes the Bunch-Kaufman factorization of a symmetric matrix `A`,
using previously allocated workspace `ws`.
If the workspace was too small and `resize==true` it will automatically resized.
If `uplo = U`, the upper half of `A` is stored. If `uplo = L`, the lower
half is stored.

Returns `A`, overwritten by the factorization, a pivot vector `ws.ipiv`, and
the error code `info` which is a non-negative integer. If `info` is positive
the matrix is singular and the diagonal part of the factorization is exactly
zero at position `info`.
"""
sytrf!(ws::BunchKaufmanWs, uplo::AbstractChar, A::AbstractMatrix; kwargs...)

"""
    sytrf_rook!(ws, uplo, A; resize=true) -> (A, ws.ipiv, info)

Similar to [`sytrf!`](@ref) but using the bounded ("rook") diagonal pivoting method.
"""
sytrf_rook!(ws::BunchKaufmanWs, uplo::AbstractChar, A::AbstractMatrix; kwargs...)

"""
    hetrf!(ws, uplo, A; resize=true) -> (A, ws.ipiv, info)

Similar as [`sytrf!`](@ref) but for Hermitian matrices.
"""
hetrf!(ws::BunchKaufmanWs, uplo::AbstractChar, A::AbstractMatrix; kwargs...)

"""
    hetrf_rook!(ws, uplo, A; resize=true) -> (A, ws.ipiv, info)

Similar to [`hetrf!`](@ref) but using the bounded ("rook") diagonal pivoting method.
"""
hetrf_rook!(ws::BunchKaufmanWs, uplo::AbstractChar, A::AbstractMatrix; kwargs...)
