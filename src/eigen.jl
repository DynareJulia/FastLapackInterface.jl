# Workspaces for Eigenvalue decomposition
using LinearAlgebra.LAPACK: chkfinite
import LinearAlgebra.LAPACK: geevx!

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
    scale::Vector{RT}
    iwork::Vector{BlasInt}
    rconde::Vector{RT}
    rcondv::Vector{RT}
end

function Base.show(io::IO, mime::MIME{Symbol("text/plain")}, ws::EigenWs)
    summary(io, ws)
    println(io)
    for f in fieldnames(EigenWs)
        print(io, "$f: ")
        summary(io, getfield(ws, f))
    end
end

# fns = fieldnames(EigenWs)
# @eval Base.iterate(ws::EigenWs) = (getfield(ws, $(fns[1])), Val(fns[2]))
# for i in 2:length(fns) - 1
#     f = fns[i]
#     @eval Base.iterate(ws::EigenWs, ::Val{$f}) = (getfield(ws, $f), Val(:($(fns[i+1]))))
# end
# @eval Base.iterate(ws::EigenWs, ::Val{$(fns[end])}) = (getfield(ws, $(fns[end])), Val(:done))
# Base.iterate(::EigenWs, ::Val{:done})    = nothing
Base.iterate(ws::EigenWs)                = (ws.work, Val(:rwork))
Base.iterate(ws::EigenWs, ::Val{:rwork}) = (ws.rwork, Val(:VL))
Base.iterate(ws::EigenWs, ::Val{:VL})    = (ws.VL, Val(:VR))
Base.iterate(ws::EigenWs, ::Val{:VR})    = (ws.VR, Val(:W))
Base.iterate(ws::EigenWs, ::Val{:W})     = (ws.W, Val(:scale))
Base.iterate(ws::EigenWs, ::Val{:scale}) = (ws.scale, Val(:iwork))
Base.iterate(ws::EigenWs, ::Val{:iwork}) = (ws.iwork, Val(:rconde))
Base.iterate(ws::EigenWs, ::Val{:rconde}) = (ws.rconde, Val(:rcondv))
Base.iterate(ws::EigenWs, ::Val{:rcondv}) = (ws.rcondv, Val(:done))
Base.iterate(::EigenWs, ::Val{:done})    = nothing

# (GE) general matrices eigenvalue-eigenvector and singular value decompositions
for (geevx, elty, relty) in
    ((:dgeevx_,:Float64,:Float64),
     (:sgeevx_,:Float32,:Float32),
     (:zgeevx_,:ComplexF64,:Float64),
     (:cgeevx_,:ComplexF32,:Float32))
    @eval begin
        # function EigenWs(A::AbstractMatrix{$elty}; lvecs = false, rvecs = true)
        #     chkstride1(A)
        #     n = checksquare(A)
        #     chkfinite(A) # balancing routines don't support NaNs and Infs
        #     VL    = similar(A, $elty, (n, lvecs ? n : 0))
        #     VR    = similar(A, $elty, (n, rvecs ? n : 0))
        #     cmplx = eltype(A) <: Complex
        #     if cmplx
        #         W     = similar(A, $elty, n)
        #         rwork = similar(A, $relty, 2n)
        #     else
        #         W     = similar(A, $elty, n)
        #         rwork = similar(A, $elty, n)
        #     end
        #     work  = Vector{$elty}(undef, 1)
        #     lwork = BlasInt(-1)
        #     info  = Ref{BlasInt}()
        #     jobvl = lvecs ? 'V' : 'N'
        #     jobvr = rvecs ? 'V' : 'N'
        #     if cmplx
        #         ccall((@blasfunc($geev), liblapack), Cvoid,
        #               (Ref{UInt8}, Ref{UInt8}, Ref{BlasInt}, Ptr{$elty},
        #                Ref{BlasInt}, Ptr{$elty}, Ptr{$elty}, Ref{BlasInt},
        #                Ptr{$elty}, Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt},
        #                Ptr{$relty}, Ptr{BlasInt}, Clong, Clong),
        #               jobvl, jobvr, n, A, max(1,stride(A,2)), W, VL, n, VR, n,
        #               work, lwork, rwork, info, 1, 1)
        #     else
        #         ccall((@blasfunc($geev), liblapack), Cvoid,
        #               (Ref{UInt8}, Ref{UInt8}, Ref{BlasInt}, Ptr{$elty},
        #                Ref{BlasInt}, Ptr{$elty}, Ptr{$elty}, Ptr{$elty},
        #                Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt}, Ptr{$elty},
        #                Ref{BlasInt}, Ptr{BlasInt}, Clong, Clong),
        #               jobvl, jobvr, n, A, max(1,stride(A,2)), W, rwork, VL, n,
        #               VR, n, work, lwork, info, 1, 1)
        #     end
        #     chklapackerror(info[])
        #     resize!(work, BlasInt(real(work[1])))
        #     return EigenWs(work, rwork, VL, VR, W)
        # end
            
        # function geev!(jobvl::AbstractChar, jobvr::AbstractChar, A::AbstractMatrix{$elty}, ws::EigenWs{$elty})
        #     chkstride1(A)
        #     n = checksquare(A)
        #     chkfinite(A) # balancing routines don't support NaNs and Infs
        #     work, rwork, VL, VR, W = ws
        #     @assert jobvl != 'V' || size(VL, 2) > 0 "Workspace was created without support for left eigenvectors,\nrecreate with EigenWs(A, lvecs = true)"
        #     @assert jobvr != 'V' || size(VR, 2) > 0 "Workspace was created without support for right eigenvectors,\nrecreate with EigenWs(A, rvecs = true)"
               
        #     cmplx = eltype(A) <: Complex
        #     info  = Ref{BlasInt}()
        #     if cmplx
        #         ccall((@blasfunc($geev), liblapack), Cvoid,
        #               (Ref{UInt8}, Ref{UInt8}, Ref{BlasInt}, Ptr{$elty},
        #                Ref{BlasInt}, Ptr{$elty}, Ptr{$elty}, Ref{BlasInt},
        #                Ptr{$elty}, Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt},
        #                Ptr{$relty}, Ptr{BlasInt}, Clong, Clong),
        #               jobvl, jobvr, n, A, max(1,stride(A,2)), W, VL, n, VR, n,
        #               work, length(work), rwork, info, 1, 1)
        #     else
        #         ccall((@blasfunc($geev), liblapack), Cvoid,
        #               (Ref{UInt8}, Ref{UInt8}, Ref{BlasInt}, Ptr{$elty},
        #                Ref{BlasInt}, Ptr{$elty}, Ptr{$elty}, Ptr{$elty},
        #                Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt}, Ptr{$elty},
        #                Ref{BlasInt}, Ptr{BlasInt}, Clong, Clong),
        #               jobvl, jobvr, n, A, max(1,stride(A,2)), W, rwork, VL, n,
        #               VR, n, work, length(work), info, 1, 1)
        #     end
        #     chklapackerror(info[])
        #     return cmplx ? (W, VL, VR) : (W, rwork, VL, VR)
        # end
        
        function EigenWs(A::AbstractMatrix{$elty}; lvecs = false, rvecs = true, sense = false)
            chkstride1(A)
            n = checksquare(A)
            chkfinite(A) # balancing routines don't support NaNs and Infs
            VL    = similar(A, $elty, (lvecs ? n : 0, n))
            VR    = similar(A, $elty, (rvecs ? n : 0, n))
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
            
            ilo = Ref{BlasInt}()
            ihi = Ref{BlasInt}()
            scale = zeros($relty, n)
            abnrm = Ref{$relty}()
            S = sense ? 'B' : 'N'
            
            rconde = zeros($relty, sense ? n : 0)
            rcondv = zeros($relty, sense ? n : 0)

            # Only needed for Float but whatever
            iwork = zeros(BlasInt, sense ? 2n - 1 : 0)
            
            if cmplx
                      
                ccall((@blasfunc($geevx), liblapack), Cvoid,
                      (Ref{UInt8}, Ref{UInt8}, Ref{UInt8}, Ref{UInt8},
                       Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt}, Ptr{$elty},
                       Ptr{$elty}, Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt},
                       Ptr{BlasInt}, Ptr{BlasInt}, Ptr{$relty}, Ptr{$relty},
                       Ptr{$relty}, Ptr{$relty}, Ptr{$elty}, Ref{BlasInt},
                       Ptr{$relty}, Ptr{BlasInt}, Clong, Clong, Clong, Clong),
                       'N', jobvl, jobvr, S,
                       n, A, max(1,stride(A,2)), W,
                       VL,n, VR, n,
                       ilo, ihi, scale, abnrm,
                       rconde, rcondv, work, lwork,
                       rwork, info, 1, 1, 1, 1)
                       
            else
                ccall((@blasfunc($geevx), liblapack), Cvoid,
                      (Ref{UInt8}, Ref{UInt8}, Ref{UInt8}, Ref{UInt8},
                       Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt}, Ptr{$elty},
                       Ptr{$elty}, Ptr{$elty}, Ref{BlasInt}, Ptr{$elty},
                       Ref{BlasInt}, Ptr{BlasInt}, Ptr{BlasInt}, Ptr{$elty},
                       Ptr{$elty}, Ptr{$elty}, Ptr{$elty}, Ptr{$elty},
                       Ref{BlasInt}, Ptr{BlasInt}, Ptr{BlasInt},
                       Clong, Clong, Clong, Clong),
                       'N', jobvl, jobvr, 'N',
                       n, A, max(1,stride(A,2)), W,
                       rwork, VL, n, VR,
                       n, ilo, ihi, scale,
                       abnrm, rconde, rcondv, work,
                       lwork, iwork, info,
                       1, 1, 1, 1)
            end
            chklapackerror(info[])
            resize!(work, BlasInt(real(work[1])))
            return EigenWs(work, rwork, VL, VR, W, scale, iwork, rconde, rcondv)
        end
        
        function geevx!(balanc::AbstractChar, jobvl::AbstractChar, jobvr::AbstractChar, sense::AbstractChar, A::AbstractMatrix{$elty}, ws::EigenWs)
            n = checksquare(A)
            chkfinite(A) # balancing routines don't support NaNs and Infs
            if balanc ∉ ['N', 'P', 'S', 'B']
                throw(ArgumentError("balanc must be 'N', 'P', 'S', or 'B', but $balanc was passed"))
            end
            if sense ∉ ['N','E','V','B']
                throw(ArgumentError("sense must be 'N', 'E', 'V' or 'B', but $sense was passed"))
            end
            work, rwork, VL, VR, W, scale, iwork, rconde, rcondv = ws
            ldvl = size(VL, 1)
            ldvr = size(VR, 1)
            
            @assert jobvl != 'V' || ldvl > 0 "Workspace was created without support for left eigenvectors,\nrecreate with EigenWs(A, lvecs = true)"
            @assert jobvr != 'V' || ldvr > 0 "Workspace was created without support for right eigenvectors,\nrecreate with EigenWs(A, rvecs = true)"
            
            abnrm = Ref{$relty}()
            ilo = Ref{BlasInt}()
            ihi = Ref{BlasInt}()
            info = Ref{BlasInt}()
            cmplx = eltype(A) <: Complex
            if cmplx
                ccall((@blasfunc($geevx), liblapack), Cvoid,
                      (Ref{UInt8}, Ref{UInt8}, Ref{UInt8}, Ref{UInt8},
                       Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt}, Ptr{$elty},
                       Ptr{$elty}, Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt},
                       Ptr{BlasInt}, Ptr{BlasInt}, Ptr{$relty}, Ptr{$relty},
                       Ptr{$relty}, Ptr{$relty}, Ptr{$elty}, Ref{BlasInt},
                       Ptr{$relty}, Ptr{BlasInt}, Clong, Clong, Clong, Clong),
                       balanc, jobvl, jobvr, sense,
                       n, A, max(1,stride(A,2)), W,
                       VL, max(1,ldvl), VR, max(1,ldvr),
                       ilo, ihi, scale, abnrm,
                       rconde, rcondv, work, length(work),
                       rwork, info, 1, 1, 1, 1)
                chklapackerror(info[])
                A, W, VL, VR, ilo[], ihi[], scale, abnrm[], rconde, rcondv
            else
                ccall((@blasfunc($geevx), liblapack), Cvoid,
                    (Ref{UInt8}, Ref{UInt8}, Ref{UInt8}, Ref{UInt8},
                     Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt}, Ptr{$elty},
                     Ptr{$elty}, Ptr{$elty}, Ref{BlasInt}, Ptr{$elty},
                     Ref{BlasInt}, Ptr{BlasInt}, Ptr{BlasInt}, Ptr{$elty},
                     Ptr{$elty}, Ptr{$elty}, Ptr{$elty}, Ptr{$elty},
                     Ref{BlasInt}, Ptr{BlasInt}, Ptr{BlasInt},
                     Clong, Clong, Clong, Clong),
                     balanc, jobvl, jobvr, sense,
                     n, A, max(1, stride(A,2)), W,
                     rwork, VL, max(1,ldvl), VR,
                     max(1,ldvr), ilo, ihi, scale,
                     abnrm, rconde, rcondv, work,
                     length(work), iwork, info,
                     1, 1, 1, 1)
                chklapackerror(info[])
                A, W, rwork, VL, VR, ilo[], ihi[], scale, abnrm[], rconde, rcondv
            end
        end
            
    end
end

# with ggev!
struct GeneralizedEigenWs
end
