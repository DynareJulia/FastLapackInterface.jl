# Workspaces for Eigenvalue decomposition
using LinearAlgebra.LAPACK: chkfinite
import LinearAlgebra.LAPACK: geev!

# Default with geevx! for normal, sygvd! for complex hermitian, syevr! for real hermitian
# The assumption is to always allocate VL and VR so that eigvecs can also be calculated
# TODO: kwarg to say that this is not needed, and assert on this while calling lapack
#       function with Ws that does not support eigvecs
struct EigenWs{T, MT <: AbstractMatrix{T}, RT<:AbstractFloat}
    work::Vector{T}
    rwork::Vector{RT} # Can be rwork if T <: Complex or WI if T <: Float64
    VL::MT
    VR::MT
    W::Vector{T} # Can be W or WR
end

function Base.show(io::IO, mime::MIME{Symbol("text/plain")}, ws::EigenWs)
    summary(io, ws)
    println(io)
    print(io, "work: ")
    summary(io, ws.work)
    println(io)
    print(io, "rwork: ")
    summary(io, ws.rwork)
    println(io)
    print(io, "VL: ")
    summary(io, ws.VL)
    print(io, "VR: ")
    summary(io, ws.VR)
    print(io, "W: ")
    return summary(io, ws.W)
end

Base.iterate(ws::EigenWs)                = (ws.work, Val(:rwork))
Base.iterate(ws::EigenWs, ::Val{:rwork}) = (ws.rwork, Val(:VL))
Base.iterate(ws::EigenWs, ::Val{:VL})    = (ws.VL, Val(:VR))
Base.iterate(ws::EigenWs, ::Val{:VR})    = (ws.VR, Val(:W))
Base.iterate(ws::EigenWs, ::Val{:W})     = (ws.W, Val(:done))
Base.iterate(::EigenWs, ::Val{:done})    = nothing

# (GE) general matrices eigenvalue-eigenvector and singular value decompositions
for (geev, elty, relty) in
    ((:dgeev_,:Float64,:Float64),
     (:sgeev_,:Float32,:Float32),
     (:zgeev_,:ComplexF64,:Float64),
     (:cgeev_,:ComplexF32,:Float32))
    @eval begin
        function EigenWs(A::AbstractMatrix{$elty}; lvecs = false, rvecs = true)
            chkstride1(A)
            n = checksquare(A)
            chkfinite(A) # balancing routines don't support NaNs and Infs
            VL    = similar(A, $elty, (n, lvecs ? n : 0))
            VR    = similar(A, $elty, (n, rvecs ? n : 0))
            cmplx = eltype(A) <: Complex
            if cmplx
                W     = similar(A, $elty, n)
                rwork = similar(A, $relty, 2n)
            else
                W     = similar(A, $elty, n)
                rwork = similar(A, $elty, n)
            end
            work  = Vector{$elty}(undef, 1)
            lwork = BlasInt(-1)
            info  = Ref{BlasInt}()
            jobvl = lvecs ? 'V' : 'N'
            jobvr = rvecs ? 'V' : 'N'
            if cmplx
                ccall((@blasfunc($geev), liblapack), Cvoid,
                      (Ref{UInt8}, Ref{UInt8}, Ref{BlasInt}, Ptr{$elty},
                       Ref{BlasInt}, Ptr{$elty}, Ptr{$elty}, Ref{BlasInt},
                       Ptr{$elty}, Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt},
                       Ptr{$relty}, Ptr{BlasInt}, Clong, Clong),
                      jobvl, jobvr, n, A, max(1,stride(A,2)), W, VL, n, VR, n,
                      work, lwork, rwork, info, 1, 1)
            else
                ccall((@blasfunc($geev), liblapack), Cvoid,
                      (Ref{UInt8}, Ref{UInt8}, Ref{BlasInt}, Ptr{$elty},
                       Ref{BlasInt}, Ptr{$elty}, Ptr{$elty}, Ptr{$elty},
                       Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt}, Ptr{$elty},
                       Ref{BlasInt}, Ptr{BlasInt}, Clong, Clong),
                      jobvl, jobvr, n, A, max(1,stride(A,2)), W, rwork, VL, n,
                      VR, n, work, lwork, info, 1, 1)
            end
            chklapackerror(info[])
            resize!(work, BlasInt(real(work[1])))
            return EigenWs(work, rwork, VL, VR, W)
        end
            
        function geev!(jobvl::AbstractChar, jobvr::AbstractChar, A::AbstractMatrix{$elty}, ws::EigenWs{$elty})
            chkstride1(A)
            n = checksquare(A)
            chkfinite(A) # balancing routines don't support NaNs and Infs
            work, rwork, VL, VR, W = ws
            @assert jobvl != 'V' || size(VL, 2) > 0 "Workspace was created without support for left eigenvectors,\nrecreate with EigenWs(A, lvecs = true)"
            @assert jobvr != 'V' || size(VR, 2) > 0 "Workspace was created without support for right eigenvectors,\nrecreate with EigenWs(A, rvecs = true)"
               
            cmplx = eltype(A) <: Complex
            info  = Ref{BlasInt}()
            if cmplx
                ccall((@blasfunc($geev), liblapack), Cvoid,
                      (Ref{UInt8}, Ref{UInt8}, Ref{BlasInt}, Ptr{$elty},
                       Ref{BlasInt}, Ptr{$elty}, Ptr{$elty}, Ref{BlasInt},
                       Ptr{$elty}, Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt},
                       Ptr{$relty}, Ptr{BlasInt}, Clong, Clong),
                      jobvl, jobvr, n, A, max(1,stride(A,2)), W, VL, n, VR, n,
                      work, length(work), rwork, info, 1, 1)
            else
                ccall((@blasfunc($geev), liblapack), Cvoid,
                      (Ref{UInt8}, Ref{UInt8}, Ref{BlasInt}, Ptr{$elty},
                       Ref{BlasInt}, Ptr{$elty}, Ptr{$elty}, Ptr{$elty},
                       Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt}, Ptr{$elty},
                       Ref{BlasInt}, Ptr{BlasInt}, Clong, Clong),
                      jobvl, jobvr, n, A, max(1,stride(A,2)), W, rwork, VL, n,
                      VR, n, work, length(work), info, 1, 1)
            end
            chklapackerror(info[])
            return cmplx ? (W, VL, VR) : (W, rwork, VL, VR)
        end
    end
end

# with ggev!
struct GeneralizedEigenWs
end
