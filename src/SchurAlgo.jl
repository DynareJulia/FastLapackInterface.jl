# general Schur decomposition with reordering
# adataped from ./base/linalg/lapack.jl
# Implement selet
using LinearAlgebra: checksquare

include("exceptions.jl")



# Select functions
# gees
# Original default
function schurselect(wr_::Ptr, wi_::Ptr)
    schurselect((wr, wi) -> wr^2 + wi^2 >= 1.0, wr_, wi_)
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
    schurselect((αr, αi, β) -> αr^2 + αi^2 < SCHUR_CRITERIUM * β^2, αr_, αi_, β_)
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

for (gees,  elty) in
    ((:dgees_,:Float64),
     (:sgees_,:Float32))
    @eval begin
        function GeesWs(A::AbstractMatrix{$elty})
            require_one_based_indexing(A)
            chkstride1(A)
            n = checksquare(A)
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
            return GeesWs{$elty}(work, info, wr, wi, vs, Ref{BlasInt}(), Vector{BlasInt}(undef, n), similar(A, Complex{$elty}, n)) 
        end
        #     .. Scalar Arguments ..
        #     CHARACTER          JOBVS, SORT
        #     INTEGER            INFO, LDA, LDVS, LWORK, N, SDIM
        #     ..
        #     .. Array Arguments ..
        #     LOGICAL            BWORK( * )
        #     DOUBLE PRECISION   A( LDA, * ), VS( LDVS, * ), WI( * ), WORK( * ),
        #    $                   WR( * )
        function gees!(jobvs::AbstractChar, A::AbstractMatrix{$elty}, ws::GeesWs{$elty})
            require_one_based_indexing(A)
            chkstride1(A)
            n     = checksquare(A)
            @assert n <= length(ws) "Allocated Workspace too small."
            work, info, wr, wi, vs, _, __ = ws
            ldvs  = max(size(vs, 1), 1)
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
        
        #     .. Scalar Arguments ..
        #     CHARACTER          JOBVS, SORT
        #     INTEGER            INFO, LDA, LDVS, LWORK, N, SDIM
        #     ..
        #     .. Array Arguments ..
        #     LOGICAL            BWORK( * )
        #     DOUBLE PRECISION   A( LDA, * ), VS( LDVS, * ), WI( * ), WORK( * ),
        #    $                   WR( * )
        function gees!(select_func::Function, jobvs::AbstractChar, A::AbstractMatrix{$elty}, ws::GeesWs{$elty})
            require_one_based_indexing(A)
            chkstride1(A)
            n     = checksquare(A)
            @assert n <= length(ws) "Allocated Workspace too small."
            work, info, wr, wi, vs, sdim, bwork = ws
            ldvs  = max(size(vs, 1), 1)
            lwork = length(work)
            sfunc(wr, wi) = schurselect(select_func, wr, wi)
            sel_func = @cfunction($(Expr(:($), sfunc)), Cint, (Ptr{Cdouble}, Ptr{Cdouble}))
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

function DgeesWs(n::Int64)
    A = zeros(n, n)
    DgeesWs(A)
end

# function dgees!(ws::DgeesWs, A::StridedMatrix{Float64})
#     n = Ref{BlasInt}(size(A, 1))
#     RldA = Ref{BlasInt}(max(1, stride(A, 2)))
#     myfunc::Function = make_select_function(>=, 1.0)
#     ccall(
#         (@blasfunc(dgees_), liblapack),
#         Cvoid,
#         (
#             Ref{UInt8},
#             Ref{UInt8},
#             Ptr{Cvoid},
#             Ref{BlasInt},
#             Ptr{Float64},
#             Ref{BlasInt},
#             Ptr{BlasInt},
#             Ptr{Float64},
#             Ptr{Float64},
#             Ptr{Float64},
#             Ref{BlasInt},
#             Ptr{Float64},
#             Ref{BlasInt},
#             Ptr{BlasInt},
#             Ptr{BlasInt},
#         ),
#         ws.jobvs,
#         'N',
#         C_NULL,
#         n,
#         A,
#         RldA,
#         ws.sdim,
#         ws.wr,
#         ws.wi,
#         ws.vs,
#         ws.ldvs,
#         ws.work,
#         ws.lwork,
#         ws.bwork,
#         ws.info,
#     )
#     copyto!(ws.eigen_values, complex.(ws.wr, ws.wi))
#     chklapackerror(ws.info[])
# end


# Make this generic

# function make_select_function(op, crit)::Function
#     mycompare = function (wr_, wi_)
#         wr = unsafe_load(wr_)
#         wi = unsafe_load(wi_)
#         return convert(Cint, op(wr * wr + wi * wi, crit) ? 1 : 0)
#     end
#     return mycompare
# end

# function dgees!(ws::DgeesWs, A::StridedMatrix{Float64}, op, crit)
#     n = Ref{BlasInt}(size(A, 1))
#     RldA = Ref{BlasInt}(max(1, stride(A, 2)))
#     myfunc::Function = make_select_function(op, crit)
#     mycompare_c = @cfunction($myfunc, Cint, (Ptr{Cdouble}, Ptr{Cdouble}))
#     ccall(
#         (@blasfunc(dgees_), liblapack),
#         Cvoid,
#         (
#             Ref{UInt8},
#             Ref{UInt8},
#             Ptr{Cvoid},
#             Ref{BlasInt},
#             Ptr{Float64},
#             Ref{BlasInt},
#             Ptr{BlasInt},
#             Ptr{Float64},
#             Ptr{Float64},
#             Ptr{Float64},
#             Ref{BlasInt},
#             Ptr{Float64},
#             Ref{BlasInt},
#             Ptr{BlasInt},
#             Ptr{BlasInt},
#         ),
#         ws.jobvs,
#         'S',
#         mycompare_c,
#         n,
#         A,
#         RldA,
#         ws.sdim,
#         ws.wr,
#         ws.wi,
#         ws.vs,
#         ws.ldvs,
#         ws.work,
#         ws.lwork,
#         ws.bwork,
#         ws.info,
#     )
#     copyto!(ws.eigen_values, complex.(ws.wr, ws.wi))
#     chklapackerror(ws.info[])
# end

mutable struct GgesWs{T}
    work::Vector{T}
    info::Ref{BlasInt}
    αr::Vector{T}
    αi::Vector{T}
    β::Vector{T}
    vsl::Matrix{T}
    vsr::Matrix{T}
    eigen_values::Vector{Complex{T}}
end

Base.length(ws::GgesWs) = length(ws.αr)
Base.iterate(ws::GgesWs)               = (ws.work, Val(:info))
Base.iterate(ws::GgesWs, ::Val{:info}) = (ws.info, Val(:αr))
Base.iterate(ws::GgesWs, ::Val{:αr})   = (ws.αr, Val(:αi))
Base.iterate(ws::GgesWs, ::Val{:αi})   = (ws.αi, Val(:β))
Base.iterate(ws::GgesWs, ::Val{:β})    = (ws.β, Val(:vsl))
Base.iterate(ws::GgesWs, ::Val{:vsl})  = (ws.vsl, Val(:vsr))
Base.iterate(ws::GgesWs, ::Val{:vsr})  = (ws.vsr, Val(:done))
Base.iterate(::GgesWs, ::Val{:done})   = nothing

# look into matlab function
for (gges, elty) in ((:dgges_,:Float64), (:sgges_,:Float32))
    @eval begin
        function GgesWs(A::AbstractMatrix{$elty})
            chkstride1(A)
            n = checksquare(A)
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
                    n, A, max(1,stride(A, 2)), A,
                    max(1,stride(A, 2)), C_NULL, αr, αi,
                    β, vsl, n, vsr,
                    n, work, lwork, C_NULL,
                    info, 1, 1, 1)
            
           chklapackerror(info[])
           resize!(work, BlasInt(real(work[1])))
           return GgesWs(work, info, αr, αi, β, vsl, vsr, similar(A, Complex{$elty}, n))
       end
           
        # *     .. Scalar Arguments ..
        #       CHARACTER          JOBVSL, JOBVSR, SORT
        #       INTEGER            INFO, LDA, LDB, LDVSL, LDVSR, LWORK, N, SDIM
        # *     ..
        # *     .. Array Arguments ..
        #       LOGICAL            BWORK( * )
        #       DOUBLE PRECISION   A( LDA, * ), ALPHAI( * ), ALPHAR( * ),
        #      $                   B( LDB, * ), BETA( * ), VSL( LDVSL, * ),
        #      $                   VSR( LDVSR, * ), WORK( * )
        function gges!(jobvsl::AbstractChar, jobvsr::AbstractChar, A::AbstractMatrix{$elty}, B::AbstractMatrix{$elty}, ws::GgesWs{$elty})
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
                n, A, max(1,stride(A, 2)), B,
                max(1,stride(B, 2)), Ref{BlasInt}(), αr, αi,
                β, vsl, ldvsl, vsr,
                ldvsr, work, length(work), C_NULL,
                info, 1, 1, 1)
            chklapackerror(info[])
            @inbounds for i in axes(A, 1)
                ws.eigen_values[i] = complex(ws.αr[i], ws.αi[i])
            end
            A, B, ws.eigen_values, ws.β, view(vsl, 1:(jobvsl == 'V' ? n : 0), :), view(vsr, 1:(jobvsr == 'V' ? n : 0),:)
        end
    end
end


# function DggesWs(A::StridedMatrix{Float64}, B::StridedMatrix{Float64})
#     DggesWs(Ref{UInt8}('N'), Ref{UInt8}('N'), Ref{UInt8}('N'), A, B)
# end

# function dgges!(
#     jobvsl::Char,
#     jobvsr::Char,
#     A::StridedMatrix{Float64},
#     B::StridedMatrix{Float64},
#     vsl::Matrix{Float64},
#     vsr::Matrix{Float64},
#     eigval::Array{ComplexF64,1},
#     ws::DggesWs,
# )
#     n = size(A, 1)
#     ldvsl = jobvsl == 'V' ? n : 1
#     ldvsr = jobvsr == 'V' ? n : 1
#     sort = 'S'
#     sdim = Ref{BlasInt}(0)
#     info = Ref{BlasInt}(0)
#     mycompare_g_c = @cfunction(mycompare, Cint, (Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}))
#     ccall((@blasfunc(dgges_), liblapack), Cvoid,
#         (Ref{UInt8}, Ref{UInt8}, Ref{UInt8}, Ptr{Cvoid}, Ref{BlasInt}, Ptr{Float64}, Ref{BlasInt}, Ptr{Float64}, Ref{BlasInt}, Ptr{BlasInt}, Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Ref{BlasInt}, Ptr{Float64}, Ref{BlasInt}, Ptr{Float64}, Ref{BlasInt}, Ptr{Int64}, Ref{Int64}),
#         jobvsl, jobvsr, sort, mycompare_g_c, n, A, max(1, stride(A, 2)), B, max(1, stride(B, 2)), sdim, ws.alphar, ws.alphai, ws.beta, vsl, ldvsl, vsr, ldvsr, ws.work, ws.lwork, ws.bwork, info)
#     ws.sdim = sdim[]
#     if info[] > 0
#         throw(DggesException(info[]))
#     end
#     for i = 1:n
#         eigval[i] = complex(ws.alphar[i], ws.alphai[i]) / ws.beta[i]
#     end
# end
