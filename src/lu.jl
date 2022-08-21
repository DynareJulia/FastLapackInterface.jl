import LinearAlgebra.LAPACK: getrf!
"""
    LUWs

Workspace to be used with the [`LinearAlgebra.LU`](https://docs.julialang.org/en/v1/stdlib/LinearAlgebra/#LinearAlgebra.LU)
representation of the LU factorization which uses the [`LAPACK.getrf!`](@ref) function.

# Examples
```jldoctest
julia> A = [1.2 2.3
            6.2 3.3]
2×2 Matrix{Float64}:
 1.2  2.3
 6.2  3.3

julia> ws = LUWs(A)
LUWs
  ipiv: 2-element Vector{Int64}

julia> t = LU(LAPACK.getrf!(ws, A)...)
LU{Float64, Matrix{Float64}, Vector{Int64}}
L factor:
2×2 Matrix{Float64}:
 1.0       0.0
 0.193548  1.0
U factor:
2×2 Matrix{Float64}:
 6.2  3.3
 0.0  1.66129
```
"""
struct LUWs <: Workspace
    ipiv::Vector{BlasInt}
end
LUWs(n::Int) = LUWs(zeros(BlasInt, n))
LUWs(a::AbstractMatrix) = LUWs(min(size(a)...))
function Base.resize!(ws::LUWs, A::AbstractMatrix)
    resize!(ws.ipiv, min(size(A)...))
    return ws
end

for (getrf, elty) in ((:dgetrf_, :Float64),
                      (:sgetrf_, :Float32),
                      (:zgetrf_, :ComplexF64),
                      (:cgetrf_, :ComplexF32))
    @eval begin
        function getrf!(ws::LUWs, A::AbstractMatrix{$elty}; resize=true)
            nws = length(ws.ipiv)
            n = min(size(A)...)
            if n != nws 
                if resize
                    resize!(ws, A)
                else
                    throw(WorkspaceSizeError(nws, n))
                end
            end
            require_one_based_indexing(A)
            chkstride1(A)
            m, n = size(A)
            lda  = max(1, stride(A, 2))
            info = Ref{BlasInt}()
            ccall((@blasfunc($getrf), liblapack), Cvoid,
                  (Ref{BlasInt}, Ref{BlasInt}, Ptr{$elty},
                   Ref{BlasInt}, Ptr{BlasInt}, Ptr{BlasInt}),
                  m, n, A, lda, ws.ipiv, info)
            chkargsok(info[])
            return A, ws.ipiv, info[] #Error code is stored in LU factorization type
        end
    end
end

"""
    getrf!(ws, A; resize=true) -> (A, ws.ipiv, info)

Compute the pivoted `LU` factorization of `A`, `A = LU`, using the preallocated [`LUWs`](@ref) workspace `ws`. If the workspace is too small and `resize==true` it will be resized
appropriately for `A`.

Returns `A`, modified in-place, `ws.ipiv`, the pivoting information, and the `ws.info`
code which indicates success (`info = 0`), a singular value in `U`
(`info = i`, in which case `U[i,i]` is singular), or an error code (`info < 0`).
"""
getrf!(ws::LUWs, A::AbstractMatrix; kwargs...)
