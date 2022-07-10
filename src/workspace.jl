"""
    Workspace(lapack_function, A)

Will create the correct [`Workspace`](@ref WorkSpaces) for the target `lapack_function` and matrix `A`.
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

function Base.show(io::IO, mime::MIME{Symbol("text/plain")}, ws::T) where {T<:Workspace}
    summary(io, ws)
    println(io)
    for f in fieldnames(T)
        print(io, "\t$f: ")
        summary(io, getfield(ws, f))
        println(io)
    end
end

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

"""
    factorize!(ws, args...)

Alias for [`decompose!`](@ref).
"""
const factorize! = decompose!
