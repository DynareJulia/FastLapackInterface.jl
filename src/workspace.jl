function Base.show(io::IO, mime::MIME{Symbol("text/plain")}, ws::T) where {T<:Workspace}
    summary(io, ws)
    println(io)
    for f in fieldnames(T)
        print(io, "  $f: ")
        summary(io, getfield(ws, f))
        println(io)
    end
end

"""
    Workspace(lapack_function, A)

Will create the correct [`Workspace`](@ref WorkSpaces) for the target `lapack_function` and matrix `A`.

# Examples
```jldoctest
julia> A = [1.2 2.3
            6.2 3.3]
2×2 Matrix{Float64}:
 1.2  2.3
 6.2  3.3

julia> ws = Workspace(LAPACK.geqrt!, A)
QRWYWs{Float64, Matrix{Float64}}
  work: 4-element Vector{Float64}
  T: 2×2 Matrix{Float64}


julia> LinearAlgebra.QRCompactWY(factorize!(ws, A)...)
QRCompactWY{Float64, Matrix{Float64}, Matrix{Float64}}
Q factor:
2×2 QRCompactWYQ{Float64, Matrix{Float64}, Matrix{Float64}}:
 -0.190022  -0.98178
 -0.98178    0.190022
R factor:
2×2 Matrix{Float64}:
 -6.31506  -3.67692
  0.0      -1.63102
```
"""
Workspace(::typeof(LAPACK.getrf!), A::AbstractMatrix) = LUWs(A)

Workspace(::typeof(LAPACK.geqrf!), A::AbstractMatrix) = QRWs(A)
Workspace(::typeof(LAPACK.ormqr!), A::AbstractMatrix) = QRWs(A)
Workspace(::typeof(LAPACK.geqrt!), A::AbstractMatrix) = QRWYWs(A)
Workspace(::typeof(LAPACK.geqp3!), A::AbstractMatrix) = QRPivotedWs(A)

Workspace(::typeof(LAPACK.gees!), A::AbstractMatrix) = SchurWs(A)
Workspace(::typeof(LAPACK.gges!), A::AbstractMatrix) = GeneralizedSchurWs(A)

Workspace(::typeof(LAPACK.geevx!), A::AbstractMatrix; kwargs...) = EigenWs(A; kwargs...)
Workspace(::typeof(LAPACK.syevr!), A::AbstractMatrix; kwargs...) = HermitianEigenWs(A; kwargs...)
Workspace(::typeof(LAPACK.ggev!), A::AbstractMatrix; kwargs...) = GeneralizedEigenWs(A; kwargs...)

Workspace(::typeof(LAPACK.sytrf!), A::AbstractMatrix) = BunchKaufmanWs(A)
Workspace(::typeof(LAPACK.sytrf_rook!), A::AbstractMatrix) = BunchKaufmanWs(A)
Workspace(::typeof(LAPACK.hetrf!), A::AbstractMatrix) = BunchKaufmanWs(A)
Workspace(::typeof(LAPACK.hetrf_rook!), A::AbstractMatrix) = BunchKaufmanWs(A)

Workspace(::typeof(LAPACK.pstrf!), A::AbstractMatrix) = CholeskyPivotedWs(A)

Workspace(::typeof(LAPACK.gglse!), A::AbstractMatrix) = LSEWs(A)

"""
    decompose!(ws, args...)

Will use the previously created [`Workspace`](@ref WorkSpaces) `ws` to dispatch to the correct LAPACK call.  
"""
decompose!(ws::LUWs, args...; kwargs...) = LAPACK.getrf!(ws, args...; kwargs...)

decompose!(ws::QRWs, args...; kwargs...)   = LAPACK.geqrf!(ws, args...; kwargs...)
decompose!(ws::QRWYWs, args...; kwargs...) = LAPACK.geqrt!(ws, args...; kwargs...)
decompose!(ws::QRPivotedWs, args...; kwargs...)  = LAPACK.geqp3!(ws, args...; kwargs...)

decompose!(ws::SchurWs, args...; kwargs...)            = LAPACK.gees!(ws, args...; kwargs...)
decompose!(ws::GeneralizedSchurWs, args...; kwargs...) = LAPACK.gges!(ws, args...; kwargs...)

decompose!(ws::EigenWs, args...; kwargs...) = LAPACK.geevx!(ws, args...; kwargs...)
decompose!(ws::HermitianEigenWs, args...; kwargs...) = LAPACK.syevr!(ws, args...; kwargs...)
decompose!(ws::GeneralizedEigenWs, args...; kwargs...) = LAPACK.ggev!(ws, args...; kwargs...)

function decompose!(ws::BunchKaufmanWs, uplo::AbstractChar, A::AbstractMatrix; rook=false, kwargs...)
    if issymmetric(A)
        return rook ? LAPACK.sytrf_rook!(ws, uplo, A; kwargs...) : LAPACK.sytrf!(ws, uplo, A; kwargs...)
    else
        return rook ? LAPACK.hetrf_rook!(ws, uplo, A; kwargs...) : LAPACK.hetrf!(ws, uplo, A; kwargs...)
    end
end

function decompose!(ws::BunchKaufmanWs, A::Hermitian; rook=false, kwargs...)
    return rook ? LAPACK.hetrf_rook!(ws, A.uplo, A.data; kwargs...) : LAPACK.hetrf!(ws, A.uplo, A.data; kwargs...)
end
function decompose!(ws::BunchKaufmanWs, A::Symmetric; rook=false, kwargs...)
    return rook ? LAPACK.sytrf_rook!(ws, A.uplo, A.data; kwargs...) : LAPACK.sytrf!(ws, A.uplo, A.data; kwargs...)
end

function decompose!(ws::CholeskyPivotedWs, uplo::AbstractChar, A::AbstractMatrix, tol=1e-16; kwargs...)
    return LAPACK.pstrf!(ws, uplo, A, tol; kwargs...)
end

function decompose!(ws::CholeskyPivotedWs, A::Union{Hermitian, Symmetric}, tol=1e-16; kwargs...)
    return LAPACK.pstrf!(ws, A.uplo, A.data, tol; kwargs...)
end

decompose!(ws::LSEWs, args...; kwargs...) = LAPACK.gglse!(ws, args...; kwargs...)

"""
    factorize!(ws, args...)

Alias for [`decompose!`](@ref).
"""
const factorize! = decompose!

"""
    resize!(ws, A; kwargs...)

Resizes the `ws` to be appropriate for use with matrix `A`. The `kwargs` can be used to
communicate which features should be supported by the [`Workspace`](@ref), such as
left and right eigenvectors while using [`EigenWs`](@ref).
This function is mainly used for automatic resizing inside [`LAPACK functions`](@ref LAPACK).
"""
Base.resize!(ws::Workspace, A::AbstractMatrix; kwargs...)
