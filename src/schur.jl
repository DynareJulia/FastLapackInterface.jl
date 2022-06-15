# general Schur decomposition with reordering

# SELECT functions
# gees
# Original default
gees_default_select() = (wr, wi) -> wr^2 + wi^2 >= 1.0
schurselect(wr_::Ptr, wi_::Ptr) = schurselect(gees_default_select, wr_, wi_)

# Generic
function schurselect(f::Function, wr_::Ptr, wi_::Ptr)
    wr = unsafe_load(wr_)
    wi = unsafe_load(wi_)
    return convert(Cint, f(wr, wi) ? 1 : 0)
end

# gges
const SCHUR_CRITERIUM = 1 + 1e-6

# Original default
gges_default_select() = (αr, αi, β) -> αr^2 + αi^2 < SCHUR_CRITERIUM * β^2
schurselect(αr_::Ptr, αi_::Ptr, β_::Ptr) = schurselect(gges_default_select, αr_, αi_, β_)

# Generic
function schurselect(f::Function, αr_::Ptr, αi_::Ptr, β_::Ptr)
    αr = unsafe_load(αr_)
    αi = unsafe_load(αi_)
    β  = unsafe_load(β_)
    return convert(Cint, f(αr, αi, β) ? 1 : 0)
end

"""
    GeesWs

Workspace to be used with the [`LinearAlgebra.Schur`](@ref) representation
of the Schur decomposition which uses the [`gees!`](@ref) LAPACK function.
Upon initialization with a template, work buffers will be allocated and stored which
will be (re)used during the factorization.

# Examples
```jldoctest
julia> A = [1.2 2.3
            6.2 3.3]
2×2 Matrix{Float64}:
 1.2  2.3
 6.2  3.3

julia> ws = GeesWs(A)
GeesWs{Float64}
work: 68-element Vector{Float64}
vs: 2×2 Matrix{Float64}
eigen_values: 2-element Vector{ComplexF64}

julia> t = Schur(LAPACK.gees!('V', A, ws)...)
Schur{Float64, Matrix{Float64}, Vector{Float64}}
T factor:
2×2 Matrix{Float64}:
 -1.6695  -3.9
  0.0      6.1695
Z factor:
2×2 Matrix{Float64}:
 -0.625424  -0.780285
  0.780285  -0.625424
eigenvalues:
2-element Vector{Float64}:
 -1.6695025194532018
  6.169502519453203

julia> Matrix(t)
2×2 Matrix{Float64}:
 1.2  2.3
 6.2  3.3
```
"""
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

function Base.show(io::IO, mime::MIME{Symbol("text/plain")}, ws::GeesWs)
    summary(io, ws); println(io)
    print(io, "work: ")
    summary(io, ws.work); println(io)
    print(io, "vs: ")
    summary(io, ws.vs); println(io)
    print(io, "eigen_values: ")
    summary(io, ws.eigen_values)
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

for (gees, elty) in ((:dgees_, :Float64),
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

        function LAPACK.gees!(jobvs::AbstractChar, A::AbstractMatrix{$elty}, ws::GeesWs{$elty})
            require_one_based_indexing(A)
            chkstride1(A)
            n = checksquare(A)
            if n > length(ws)
                throw(ArgumentError("Allocated workspace has length $(length(ws)), but needs length $n."))
            end
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

        function LAPACK.gees!(select_func::Function, jobvs::AbstractChar, A::AbstractMatrix{$elty},
                       ws::GeesWs{$elty})
            require_one_based_indexing(A)
            chkstride1(A)
            n = checksquare(A)
            if n > length(ws)
                throw(ArgumentError("Allocated workspace has length $(length(ws)), but needs length $n."))
            end
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

"""
    gees!([select], jobvs, A, ws) -> (A, vs, ws.eigen_values)

Computes the eigenvalues (`jobvs = N`) or the eigenvalues and Schur
vectors (`jobvs = V`) of matrix `A`, using the preallocated [`GeesWs`](@ref) worspace `ws`.
`A` is overwritten by its Schur form, and `ws.eigen_values` is overwritten with the eigenvalues.

It is possible to specify `select`, a function used to sort the eigenvalues during the decomponsition.
The function should accept have the signature `f(wr::Float64, wi::Float64) -> Bool`, where
`wr` and `wi` are the real and imaginary parts of the eigenvalue. 

Returns `A`, `vs` containing the Schur vectors, and `ws.eigen_values`.
"""
LAPACK.gees!(jobvs::AbstractChar, A::AbstractMatrix, ws::GeesWs)


"""
    GgesWs

Workspace to be used with the [`LinearAlgebra.GeneralizedSchur`](@ref) representation
of the Generalized Schur decomposition which uses the [`gges!`](@ref) LAPACK function.
Upon initialization with a template, work buffers will be allocated and stored which
will be (re)used during the factorization.

# Examples
```jldoctest
julia> A = [1.2 2.3
            6.2 3.3]
2×2 Matrix{Float64}:
 1.2  2.3
 6.2  3.3

julia> B = [8.2 0.3
            1.7 4.3]
2×2 Matrix{Float64}:
 8.2  0.3
 1.7  4.3

julia> ws = GgesWs(A)
GgesWs{Float64}
work: 90-element Vector{Float64}
vsl: 2×2 Matrix{Float64}
vsr: 2×2 Matrix{Float64}
eigen_values: 2-element Vector{ComplexF64}

julia> t = GeneralizedSchur(LAPACK.gges!('V','V', A, B, ws)...)
GeneralizedSchur{Float64, Matrix{Float64}, Vector{ComplexF64}, Vector{Float64}}
S factor:
2×2 Matrix{Float64}:
 -1.43796  1.63843
  0.0      7.16295
T factor:
2×2 Matrix{Float64}:
 5.06887  -4.00221
 0.0       6.85558
Q factor:
2×2 Matrix{Float64}:
 -0.857329  0.514769
  0.514769  0.857329
Z factor:
2×2 Matrix{Float64}:
 -0.560266  0.828313
  0.828313  0.560266
α:
2-element Vector{ComplexF64}:
 -1.4379554610733563 + 0.0im
   7.162947865097022 + 0.0im
β:
2-element Vector{Float64}:
 5.068865029631368
 6.855578082442485
```
"""
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

function Base.show(io::IO, mime::MIME{Symbol("text/plain")}, ws::GgesWs)
    summary(io, ws); println(io)
    print(io, "work: ")
    summary(io, ws.work); println(io)
    print(io, "vsl: ")
    summary(io, ws.vsl); println(io)
    print(io, "vsr: ")
    summary(io, ws.vsr); println(io)
    print(io, "eigen_values: ")
    summary(io, ws.eigen_values)
end

Base.length(ws::GgesWs) = length(ws.αr)

Base.iterate(ws::GgesWs)                = (ws.work, Val(:info))
Base.iterate(ws::GgesWs, ::Val{:info})  = (ws.info, Val(:αr))
Base.iterate(ws::GgesWs, ::Val{:αr})    = (ws.αr, Val(:αi))
Base.iterate(ws::GgesWs, ::Val{:αi})    = (ws.αi, Val(:β))
Base.iterate(ws::GgesWs, ::Val{:β})     = (ws.β, Val(:vsl))
Base.iterate(ws::GgesWs, ::Val{:vsl})   = (ws.vsl, Val(:vsr))
Base.iterate(ws::GgesWs, ::Val{:vsr})   = (ws.vsr, Val(:sdim))
Base.iterate(ws::GgesWs, ::Val{:sdim})  = (ws.sdim, Val(:bwork))
Base.iterate(ws::GgesWs, ::Val{:bwork}) = (ws.bwork, Val(:done))
Base.iterate(::GgesWs, ::Val{:done})    = nothing

# look into matlab function
for (gges, elty) in ((:dgges_, :Float64),
                     (:sgges_, :Float32))
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

        function LAPACK.gges!(jobvsl::AbstractChar, jobvsr::AbstractChar, A::AbstractMatrix{$elty},
                       B::AbstractMatrix{$elty}, ws::GgesWs{$elty})
            chkstride1(A, B)
            n, m = checksquare(A, B)
            if n != m
                throw(DimensionMismatch("dimensions of A, ($n,$n), and B, ($m,$m), must match"))
            end
            if n > length(ws)
                throw(ArgumentError("Allocated workspace has length $(length(ws)), but needs length $n."))
            end

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

        function LAPACK.gges!(select_func::Function, jobvsl::AbstractChar, jobvsr::AbstractChar,
                       A::AbstractMatrix{$elty}, B::AbstractMatrix{$elty},
                       ws::GgesWs{$elty})
            chkstride1(A, B)
            n, m = checksquare(A, B)
            if n != m
                throw(DimensionMismatch("dimensions of A, ($n,$n), and B, ($m,$m), must match"))
            end
            if n > length(ws)
                throw(ArgumentError("Allocated workspace has length $(length(ws)), but needs length $n."))
            end

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

"""
    gges!([select], jobvsl, jobvsr, A, B, ws) -> (A, B, ws.eigen_values, ws.β, ws.vsl, ws.vsr)

Computes the generalized eigenvalues, generalized Schur form, left Schur
vectors (`jobsvl = V`), or right Schur vectors (`jobvsr = V`) of `A` and
`B`, using preallocated [`GgesWs`](@ref) workspace `ws`.

It is possible to specify `select`, a function used to sort the eigenvalues during the decomponsition.
The function should accept have the signature `f(αr::Float64, αi::Float64, β::Float64) -> Bool`, where
`αr` and `αi` are the real and imaginary parts of the eigenvalue, and `β` the factor. 

The generalized eigenvalues are returned in `ws.eigen_values` and `ws.β`. The left Schur
vectors are returned in `ws.vsl` and the right Schur vectors are returned in `ws.vsr`.
"""
LAPACK.gges!(jobvsl::AbstractChar, jobvsr::AbstractChar, A::AbstractMatrix, B::AbstractMatrix)
