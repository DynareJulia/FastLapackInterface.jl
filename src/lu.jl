"""
    LUWs

Workspace to be used with the [`LinearAlgebra.LU`](@ref) representation
of the LU factorization which uses the [`getrf!`](@ref) LAPACK function.
Upon initialization with a template, work buffers will be allocated and stored which
will be (re)used during the factorization.

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

julia> t = LU(LAPACK.getrf!(A, ws)...)
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
struct LUWs
    ipiv::Vector{BlasInt}
end
function Base.show(io::IO, mime::MIME{Symbol("text/plain")}, ws::LUWs)
    summary(io, ws); println(io)
    print(io, "ipiv: ")
    summary(io, ws.ipiv)
end

LUWs(n::Int) = LUWs(zeros(BlasInt, n))
LUWs(a::AbstractMatrix) = LUWs(min(size(a)...))

for (getrf, elty) in ((:dgetrf_, :Float64),
                      (:sgetrf_, :Float32),
                      (:zgetrf_, :ComplexF64),
                      (:cgetrf_, :ComplexF32))
    @eval begin
        function LAPACK.getrf!(A::AbstractMatrix{$elty}, ws::LUWs)
            @assert min(size(A)...) <= length(ws.ipiv) "Allocated Workspace is too small."
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
# No need to reimplement the solve because can just reuse LU from base Julia

"""
    getrf!(A, ws) -> (A, ws.ipiv, ws.info)

Compute the pivoted `LU` factorization of `A`, `A = LU`, using the preallocated [`LUWs`](@ref) workspace `ws`.

Returns `A`, modified in-place, `ws.ipiv`, the pivoting information, and the `ws.info`
code which indicates success (`info = 0`), a singular value in `U`
(`info = i`, in which case `U[i,i]` is singular), or an error code (`info < 0`).
"""
LAPACK.getrf!(A::AbstractMatrix, ws::LUWs)
