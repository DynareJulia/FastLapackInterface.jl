# general Schur decomposition with reordering
using LinearAlgebra: checksquare
include("exceptions.jl")

# Select functions
# gees
# Original default
function schurselect(wr_::Ptr, wi_::Ptr)
    return schurselect((wr, wi) -> wr^2 + wi^2 >= 1.0, wr_, wi_)
end

# Generic
function schurselect(f::Function, wr_::Ptr, wi_::Ptr)
    wr = unsafe_load(wr_)
    wi = unsafe_load(wi_)
    return convert(Cint, f(wr, wi) ? 1 : 0)
end

# gges
const SCHUR_CRITERIUM = 1 + 1e-6
# Original default
function schurselect(αr_::Ptr, αi_::Ptr, β_::Ptr)
    return schurselect((αr, αi, β) -> αr^2 + αi^2 < SCHUR_CRITERIUM * β^2, αr_, αi_, β_)
end

# Generic
function schurselect(f::Function, αr_::Ptr, αi_::Ptr, β_::Ptr)
    αr = unsafe_load(αr_)
    αi = unsafe_load(αi_)
    β  = unsafe_load(β_)
    return convert(Cint, f(αr, αi, β) ? 1 : 0)
end

# Do we ever want to SELECT?
mutable struct GeesWs{T<:AbstractFloat}
    work::Vector{T}
    info::Ref{BlasInt}
    wr::Vector{T}
    wi::Vector{T}
    vs::Matrix{T}
    sdim::Ref{BlasInt}
    bwork::Vector{BlasInt}
    eigen_values::Vector{Complex{T}}
end

Base.length(ws::GeesWs) = length(ws.wr)

Base.iterate(ws::GeesWs)                = (ws.work, Val(:info))
Base.iterate(ws::GeesWs, ::Val{:info})  = (ws.info, Val(:wr))
Base.iterate(ws::GeesWs, ::Val{:wr})    = (ws.wr, Val(:wi))
Base.iterate(ws::GeesWs, ::Val{:wi})    = (ws.wi, Val(:vs))
Base.iterate(ws::GeesWs, ::Val{:vs})    = (ws.vs, Val(:sdim))
Base.iterate(ws::GeesWs, ::Val{:sdim})  = (ws.sdim, Val(:bwork))
Base.iterate(ws::GeesWs, ::Val{:bwork}) = (ws.bwork, Val(:done))
Base.iterate(::GeesWs, ::Val{:done})    = nothing

for (gees, elty) in
    ((:dgees_, :Float64),
     (:sgees_, :Float32))
    @eval begin
        function GeesWs(A::AbstractMatrix{$elty})
            require_one_based_indexing(A)
            chkstride1(A)
            n     = checksquare(A)
            wr    = similar(A, $elty, n)
            wi    = similar(A, $elty, n)
            vs    = similar(A, $elty, n, n)
            ldvs  = max(size(vs, 1), 1)
            work  = Vector{$elty}(undef, 1)
            lwork = BlasInt(-1)
            info  = Ref{BlasInt}()
            ccall((@blasfunc($gees), liblapack), Cvoid,
                  (Ref{UInt8}, Ref{UInt8}, Ptr{Cvoid}, Ref{BlasInt},
                   Ptr{$elty}, Ref{BlasInt}, Ptr{Cvoid}, Ptr{$elty},
                   Ptr{$elty}, Ptr{$elty}, Ref{BlasInt}, Ptr{$elty},
                   Ref{BlasInt}, Ptr{Cvoid}, Ptr{BlasInt}, Clong, Clong),
                  'V', 'N', C_NULL, n,
                  A, max(1, stride(A, 2)), C_NULL, wr,
                  wi, vs, ldvs, work,
                  lwork, C_NULL, info, 1, 1)

            chklapackerror(info[])

            resize!(work, BlasInt(real(work[1])))
            return GeesWs{$elty}(work, info, wr, wi, vs, Ref{BlasInt}(),
                                 Vector{BlasInt}(undef, n), similar(A, Complex{$elty}, n))
        end

        function gees!(jobvs::AbstractChar, A::AbstractMatrix{$elty}, ws::GeesWs{$elty})
            require_one_based_indexing(A)
            chkstride1(A)
            n = checksquare(A)
            @assert n <= length(ws) "Allocated Workspace too small."
            work, info, wr, wi, vs = ws
            ldvs = max(size(vs, 1), 1)
            lwork = length(work)
            ccall((@blasfunc($gees), liblapack), Cvoid,
                  (Ref{UInt8}, Ref{UInt8}, Ptr{Cvoid}, Ref{BlasInt},
                   Ptr{$elty}, Ref{BlasInt}, Ptr{BlasInt}, Ptr{$elty},
                   Ptr{$elty}, Ptr{$elty}, Ref{BlasInt}, Ptr{$elty},
                   Ref{BlasInt}, Ptr{Cvoid}, Ptr{BlasInt}, Clong, Clong),
                  jobvs, 'N', C_NULL, n,
                  A, max(1, stride(A, 2)), Ref{BlasInt}(), wr,
                  wi, vs, ldvs, work,
                  lwork, C_NULL, info, 1, 1)

            if iszero(wi)
                return A, vs, wr
            else
                @inbounds for i in axes(A, 1)
                    ws.eigen_values[i] = complex(wr[i], wi[i])
                end
                return A, vs, iszero(wi) ? wr : ws.eigen_values
            end
        end

        function gees!(select_func::Function, jobvs::AbstractChar, A::AbstractMatrix{$elty},
                       ws::GeesWs{$elty})
            require_one_based_indexing(A)
            chkstride1(A)
            n = checksquare(A)
            @assert n <= length(ws) "Allocated Workspace too small."
            work, info, wr, wi, vs, sdim, bwork = ws
            ldvs = max(size(vs, 1), 1)
            lwork = length(work)
            sfunc(wr, wi) = schurselect(select_func, wr, wi)
            sel_func = @cfunction($(Expr(:$, :sfunc)), Cint, (Ptr{Cdouble}, Ptr{Cdouble}))
            ccall((@blasfunc($gees), liblapack), Cvoid,
                  (Ref{UInt8}, Ref{UInt8}, Ptr{Cvoid}, Ref{BlasInt},
                   Ptr{$elty}, Ref{BlasInt}, Ptr{BlasInt}, Ptr{$elty},
                   Ptr{$elty}, Ptr{$elty}, Ref{BlasInt}, Ptr{$elty},
                   Ref{BlasInt}, Ptr{BlasInt}, Ptr{BlasInt}, Clong, Clong),
                  jobvs, 'S', sel_func, n,
                  A, max(1, stride(A, 2)), sdim, wr,
                  wi, vs, ldvs, work,
                  lwork, bwork, info, 1, 1)

            if iszero(wi)
                return A, vs, wr
            else
                @inbounds for i in axes(A, 1)
                    ws.eigen_values[i] = complex(wr[i], wi[i])
                end
                return A, vs, iszero(wi) ? wr : ws.eigen_values
            end
        end
    end
end

# Make this generic
mutable struct GgesWs{T}
    work::Vector{T}
    info::Ref{BlasInt}
    αr::Vector{T}
    αi::Vector{T}
    β::Vector{T}
    vsl::Matrix{T}
    vsr::Matrix{T}
    sdim::Ref{BlasInt}
    bwork::Vector{BlasInt}
    eigen_values::Vector{Complex{T}}
end

Base.length(ws::GgesWs) = length(ws.αr)
Base.iterate(ws::GgesWs) = (ws.work, Val(:info))
Base.iterate(ws::GgesWs, ::Val{:info}) = (ws.info, Val(:αr))
Base.iterate(ws::GgesWs, ::Val{:αr}) = (ws.αr, Val(:αi))
Base.iterate(ws::GgesWs, ::Val{:αi}) = (ws.αi, Val(:β))
Base.iterate(ws::GgesWs, ::Val{:β}) = (ws.β, Val(:vsl))
Base.iterate(ws::GgesWs, ::Val{:vsl}) = (ws.vsl, Val(:vsr))
Base.iterate(ws::GgesWs, ::Val{:vsr}) = (ws.vsr, Val(:sdim))
Base.iterate(ws::GgesWs, ::Val{:sdim}) = (ws.sdim, Val(:bwork))
Base.iterate(ws::GgesWs, ::Val{:bwork}) = (ws.bwork, Val(:done))
Base.iterate(::GgesWs, ::Val{:done}) = nothing

# look into matlab function
for (gges, elty) in ((:dgges_, :Float64), (:sgges_, :Float32))
    @eval begin
        function GgesWs(A::AbstractMatrix{$elty})
            chkstride1(A)
            n     = checksquare(A)
            αr    = similar(A, $elty, n)
            αi    = similar(A, $elty, n)
            β     = similar(A, $elty, n)
            vsl   = similar(A, $elty, n, n)
            vsr   = similar(A, $elty, n, n)
            work  = Vector{$elty}(undef, 1)
            lwork = BlasInt(-1)
            info  = Ref{BlasInt}()
            ccall((@blasfunc($gges), liblapack), Cvoid,
                  (Ref{UInt8}, Ref{UInt8}, Ref{UInt8}, Ptr{Cvoid},
                   Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt}, Ptr{$elty},
                   Ref{BlasInt}, Ptr{Cvoid}, Ptr{$elty}, Ptr{$elty},
                   Ptr{$elty}, Ptr{$elty}, Ref{BlasInt}, Ptr{$elty},
                   Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt}, Ptr{Cvoid},
                   Ref{BlasInt}, Clong, Clong, Clong),
                  'V', 'V', 'N', C_NULL,
                  n, A, max(1, stride(A, 2)), A,
                  max(1, stride(A, 2)), C_NULL, αr, αi,
                  β, vsl, n, vsr,
                  n, work, lwork, C_NULL,
                  info, 1, 1, 1)

            chklapackerror(info[])
            resize!(work, BlasInt(real(work[1])))
            return GgesWs(work, info, αr, αi, β, vsl, vsr, Ref{BlasInt}(),
                          Vector{BlasInt}(undef, n), similar(A, Complex{$elty}, n))
        end

        function gges!(jobvsl::AbstractChar, jobvsr::AbstractChar, A::AbstractMatrix{$elty},
                       B::AbstractMatrix{$elty}, ws::GgesWs{$elty})
            chkstride1(A, B)
            n, m = checksquare(A, B)
            if n != m
                throw(DimensionMismatch("dimensions of A, ($n,$n), and B, ($m,$m), must match"))
            end
            @assert n <= length(ws) "Allocated Workspace too small."

            work, info, αr, αi, β, vsl, vsr = ws
            ldvsl = size(vsl, 1)
            ldvsr = size(vsr, 1)
            ccall((@blasfunc($gges), liblapack), Cvoid,
                  (Ref{UInt8}, Ref{UInt8}, Ref{UInt8}, Ptr{Cvoid},
                   Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt}, Ptr{$elty},
                   Ref{BlasInt}, Ref{BlasInt}, Ptr{$elty}, Ptr{$elty},
                   Ptr{$elty}, Ptr{$elty}, Ref{BlasInt}, Ptr{$elty},
                   Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt}, Ptr{Cvoid},
                   Ref{BlasInt}, Clong, Clong, Clong),
                  jobvsl, jobvsr, 'N', C_NULL,
                  n, A, max(1, stride(A, 2)), B,
                  max(1, stride(B, 2)), Ref{BlasInt}(), αr, αi,
                  β, vsl, ldvsl, vsr,
                  ldvsr, work, length(work), C_NULL,
                  info, 1, 1, 1)
            chklapackerror(info[])
            @inbounds for i in axes(A, 1)
                ws.eigen_values[i] = complex(ws.αr[i], ws.αi[i])
            end
            return A, B, ws.eigen_values, ws.β, view(vsl, 1:(jobvsl == 'V' ? n : 0), :),
                   view(vsr, 1:(jobvsr == 'V' ? n : 0), :)
        end

        function gges!(select_func::Function, jobvsl::AbstractChar, jobvsr::AbstractChar,
                       A::AbstractMatrix{$elty}, B::AbstractMatrix{$elty},
                       ws::GgesWs{$elty})
            chkstride1(A, B)
            n, m = checksquare(A, B)
            if n != m
                throw(DimensionMismatch("dimensions of A, ($n,$n), and B, ($m,$m), must match"))
            end
            @assert n <= length(ws) "Allocated Workspace too small."

            work, info, αr, αi, β, vsl, vsr, sdim, bwork = ws
            ldvsl = size(vsl, 1)
            ldvsr = size(vsr, 1)
            sfunc(αr, αi, β) = schurselect(select_func, αr, αi, β)
            sel_func = @cfunction($(Expr(:$, :sfunc)), Cint,
                                  (Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}))
            ccall((@blasfunc($gges), liblapack), Cvoid,
                  (Ref{UInt8}, Ref{UInt8}, Ref{UInt8}, Ptr{Cvoid},
                   Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt}, Ptr{$elty},
                   Ref{BlasInt}, Ref{BlasInt}, Ptr{$elty}, Ptr{$elty},
                   Ptr{$elty}, Ptr{$elty}, Ref{BlasInt}, Ptr{$elty},
                   Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt}, Ptr{Cvoid},
                   Ref{BlasInt}, Clong, Clong, Clong),
                  jobvsl, jobvsr, 'S', sel_func,
                  n, A, max(1, stride(A, 2)), B,
                  max(1, stride(B, 2)), sdim, αr, αi,
                  β, vsl, ldvsl, vsr,
                  ldvsr, work, length(work), bwork,
                  info, 1, 1, 1)
            chklapackerror(info[])
            @inbounds for i in axes(A, 1)
                ws.eigen_values[i] = complex(ws.αr[i], ws.αi[i])
            end
            return A, B, ws.eigen_values, ws.β, view(vsl, 1:(jobvsl == 'V' ? n : 0), :),
                   view(vsr, 1:(jobvsr == 'V' ? n : 0), :)
        end
    end
end
