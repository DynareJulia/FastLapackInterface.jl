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
Workspace(::typeof(LAPACK.geqp3!), A::AbstractMatrix) = QRpWs(A)

Workspace(::typeof(LAPACK.gees!), A::AbstractMatrix) = SchurWs(A)
Workspace(::typeof(LAPACK.gges!), A::AbstractMatrix) = GeneralizedSchurWs(A)

Workspace(::typeof(LAPACK.geevx!), A::AbstractMatrix; kwargs...) = EigenWs(A; kwargs...)
Workspace(::typeof(LAPACK.syevr!), A::AbstractMatrix; kwargs...) = HermitianEigenWs(A; kwargs...)
Workspace(::typeof(LAPACK.ggev!), A::AbstractMatrix; kwargs...) = GeneralizedEigenWs(A; kwargs...)

Workspace(::typeof(LAPACK.sytrf!), A::AbstractMatrix) = BunchKaufmanWs(A)
Workspace(::typeof(LAPACK.sytrf_rook!), A::AbstractMatrix) = BunchKaufmanWs(A)

"""
    decompose!(ws, args...)

Will use the previously created [`Workspace`](@ref WorkSpaces) `ws` to dispatch to the correct LAPACK call.  
"""
decompose!(ws::LUWs, args...) = LAPACK.getrf!(ws, args...)

decompose!(ws::QRWs, args...)   = LAPACK.geqrf!(ws, args...)
decompose!(ws::QRWYWs, args...) = LAPACK.geqrt!(ws, args...)
decompose!(ws::QRpWs, args...)  = LAPACK.geqp3!(ws, args...)

decompose!(ws::SchurWs, args...)            = LAPACK.gees!(ws, args...)
decompose!(ws::GeneralizedSchurWs, args...) = LAPACK.gges!(ws, args...)

decompose!(ws::EigenWs, args...) = LAPACK.geevx!(ws, args...)
decompose!(ws::HermitianEigenWs, args...) = LAPACK.syevr!(ws, args...)
decompose!(ws::GeneralizedEigenWs, args...) = LAPACK.ggev!(ws, args...)

decompose!(ws::BunchKaufmanWs, args...) = LAPACK.sytrf!(ws, args...)

"""
    factorize!(ws, args...)

Alias for [`decompose!`](@ref).
"""
const factorize! = decompose!
