import LinearAlgebra.LAPACK: gglse!

"""
    LSEWs

Workspace for the least squares solving function [`LAPACK.geqrf!`](@ref).

# Examples
```jldoctest
julia> A = [1.2 2.3 6.2
            6.2 3.3 8.8
            9.1 2.1 5.5]
3×3 Matrix{Float64}:
 1.2  2.3  6.2
 6.2  3.3  8.8
 9.1  2.1  5.5

julia> B = [2.7 3.1 7.7
            4.1 8.1 1.8]
2×3 Matrix{Float64}:
 2.7  3.1  7.7
 4.1  8.1  1.8

julia> c = [0.2, 7.2, 2.9]
3-element Vector{Float64}:
 0.2
 7.2
 2.9

julia> d = [3.9, 2.1]
2-element Vector{Float64}:
 3.9
 2.1

julia> ws = LSEWs(A, B)
LSEWs{Float64}
  work: 101-element Vector{Float64}
  X: 3-element Vector{Float64}

julia> LAPACK.gglse!(ws, A, c, B, d)
([0.19723156207005318, 0.0683561362406917, 0.40981438442398854], 13.750943845251626)
```
"""
struct LSEWs{T} <: Workspace
    work::Vector{T}
    X::Vector{T}
end

LSEWs(A::AbstractMatrix) = LSEWs(A, A)
LSEWs(A::AbstractMatrix, B::AbstractMatrix) = resize!(LSEWs(Vector{eltype(A)}(undef, 1), Vector{eltype(A)}(undef, size(A,2))), A, B)
for (gglse, elty) in ((:dgglse_, :Float64),
                      (:sgglse_, :Float32),
                      (:zgglse_, :ComplexF64),
                      (:cgglse_, :ComplexF32))
    @eval begin
        function Base.resize!(ws::LSEWs, A::AbstractMatrix{$elty}, B::AbstractMatrix{$elty}; work=true, blocksize=32)
            require_one_based_indexing(A)
            chkstride1(A)
            m, n = size(A)
            p = size(B,1)
            resize!(ws.X, n)
            if work
                resize!(ws.work, p + min(m, n) + max(m,n)*blocksize)
            end
            return ws
            
        end

        function gglse!(ws::LSEWs{$elty}, A::AbstractMatrix{$elty}, c::AbstractVector{$elty},
                        B::AbstractMatrix{$elty}, d::AbstractVector{$elty}; resize=true, blocksize=32)
            require_one_based_indexing(A, c, B, d)
            chkstride1(A, c, B, d)
            m, n = size(A)
            p = size(B, 1)
            if size(B, 2) != n
                throw(DimensionMismatch("B has second dimension $(size(B,2)), needs $n"))
            end
            if length(c) != m
                throw(DimensionMismatch("c has length $(length(c)), needs $m"))
            end
            if length(d) != p
                throw(DimensionMismatch("d has length $(length(d)), needs $p"))
            end
            if n > m + p 
                throw(DimensionMismatch("Rows of A + rows of B needs to be larger than columns of A and B."))
            end
            nws = length(ws.X)
            if nws != n
                if resize
                    resize!(ws.X, n)
                    worksize = p + min(m, n) + max(m,n)*blocksize
                    if length(ws.work) < worksize
                        resize!(ws.work, worksize)
                    end
                else
                    throw(WorkspaceSizeError(nws, n))
                end
            end
                
            info  = Ref{BlasInt}()
            ccall((@blasfunc($gglse), liblapack), Cvoid,
                  (Ref{BlasInt}, Ref{BlasInt}, Ref{BlasInt}, Ptr{$elty},
                   Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt}, Ptr{$elty},
                   Ptr{$elty}, Ptr{$elty}, Ptr{$elty}, Ref{BlasInt},
                   Ptr{BlasInt}),
                  m, n, p, A, max(1,stride(A,2)), B, max(1,stride(B,2)), c, d, ws.X,
                  ws.work, length(ws.work), info)
                chklapackerror(info[])
            ws.X, dot(view(c, n - p + 1:m), view(c, n - p + 1:m))
        end
    end
end

"""
    gglse!(ws, A, c, B, d) -> (ws.X,res)

Solves the equation `A * x = c` where `x` is subject to the equality
constraint `B * x = d`. Uses the formula `||c - A*x||^2 = 0` to solve.
Uses preallocated [`LSEWs`](@ref) to store `X` and work buffers. 
Returns `ws.X` and the residual sum-of-squares.
"""
gglse!(ws::LSEWs, A::AbstractMatrix, c::AbstractVector, B::AbstractMatrix, d::AbstractVector)
