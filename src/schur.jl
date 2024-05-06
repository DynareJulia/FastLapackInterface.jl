import LinearAlgebra.LAPACK: gees!, gges!
# general Schur decomposition with reordering

# TODO: See if SELECT functions can be optimized.

# SELECT functions
"""
   enum SCHURORDER lhpm lhpp rhpm rhpp udim udip udom udop
# Keywords
    - `lhpm`: Left-half plane (abs(eigenvalue)^2 < -ϵ)
    - `lhpp`: Left-half plane (abs(eigenvalue)^2 < ϵ)
    - `rhpm`: Right-half plane (abs(eigenvalue)^2 >= -ϵ)
    - `rhpp`: Right-half plane (abs(eigenvalue)^2 >= ϵ)
    - `udim`: Interior of unit disk (abs(eigenvalue)^2 < 1 - ϵ) 
    - `udip`: Interior of unit disk (abs(eigenvalue)^2 < 1 + ϵ) 
    - `udom`: Exterior of unit disk (abs(eigenvalue)^2 >= 1 - ϵ) 
    - `udop`: Exterior of unit disk (abs(eigenvalue)^2 >= 1 + ϵ) 

# Note
    - ϵ is used to make the ordering robust to numerical error in computing repeated eigenvalues
    - ϵ is set by FastLapackInterface.SCHUR_EPSILON (default: 1e-6)
"""
@enum SCHURORDER lhpm lhpp rhpm rhpp udim udip udom udop

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

function make_geesselect(f::Function)
    function sfunc(wr, wi)::Cint
        return schurselect(f, wr, wi)
    end
    return sfunc
end

# gges
const SCHUR_CRITERIUM = 1 + 1e-6
const SCHUR_EPSILON = 1e-6

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

function lhpmselect(wr_::Ptr, wi_::Ptr)::Cint
    wr = unsafe_load(wr_)
    wi = unsafe_load(wi_)
    return (wr^2 + wi^2 < -SCHUR_EPSILON)
end
    
function lhppselect(wr_::Ptr, wi_::Ptr)::Cint
    wr = unsafe_load(wr_)
    wi = unsafe_load(wi_)
    return (wr^2 + wi^2 < SCHUR_EPSILON)
end
    
function rhpmselect(wr_::Ptr, wi_::Ptr)::Cint
    wr = unsafe_load(wr_)
    wi = unsafe_load(wi_)
    return (wr^2 + wi^2 >= -SCHUR_EPSILON)
end
    
function rhppselect(wr_::Ptr, wi_::Ptr)::Cint
    wr = unsafe_load(wr_)
    wi = unsafe_load(wi_)
    return (wr^2 + wi^2 >= SCHUR_EPSILON)
end
    
function udimselect(wr_::Ptr, wi_::Ptr)::Cint
    wr = unsafe_load(wr_)
    wi = unsafe_load(wi_)
    return (wr^2 + wi^2 < (1 - SCHUR_EPSILON))
end
    
function udipselect(wr_::Ptr, wi_::Ptr)::Cint
    wr = unsafe_load(wr_)
    wi = unsafe_load(wi_)
    return (wr^2 + wi^2 < (1 + SCHUR_EPSILON))
end
    
function udomselect(wr_::Ptr, wi_::Ptr)::Cint
    wr = unsafe_load(wr_)
    wi = unsafe_load(wi_)
    return (wr^2 + wi^2 >= (1 - SCHUR_EPSILON))
end
    
function udopselect(wr_::Ptr, wi_::Ptr)::Cint
    wr = unsafe_load(wr_)
    wi = unsafe_load(wi_)
    return (wr^2 + wi^2 >= (1 + SCHUR_EPSILON))
end

function lhpmselect(αr_::Ptr, αi_::Ptr, β_::Ptr)::Cint
    αr = unsafe_load(αr_)
    αi = unsafe_load(αi_)
    β  = unsafe_load(β_)
    return (αr^2 + αi^2 < -SCHUR_EPSILON*β^2)
end
    
function lhppselect(αr_::Ptr, αi_::Ptr, β_::Ptr)::Cint
    αr = unsafe_load(αr_)
    αi = unsafe_load(αi_)
    β  = unsafe_load(β_)
    return (αr^2 + αi^2 < SCHUR_EPSILON*β^2)
end
    
function rhpmselect(αr_::Ptr, αi_::Ptr, β_::Ptr)::Cint
    αr = unsafe_load(αr_)
    αi = unsafe_load(αi_)
    β  = unsafe_load(β_)
    return (αr^2 + αi^2 >= -SCHUR_EPSILON*β^2)
end
    
function rhppselect(αr_::Ptr, αi_::Ptr, β_::Ptr)::Cint
    αr = unsafe_load(αr_)
    αi = unsafe_load(αi_)
    β  = unsafe_load(β_)
    return (αr^2 + αi^2 >= SCHUR_EPSILON*β^2)
end
    
function udimselect(αr_::Ptr, αi_::Ptr, β_::Ptr)::Cint
    αr = unsafe_load(αr_)
    αi = unsafe_load(αi_)
    β  = unsafe_load(β_)
    return (αr^2 + αi^2 < (1 - SCHUR_EPSILON)*β^2)
end
    
function udipselect(αr_::Ptr, αi_::Ptr, β_::Ptr)::Cint
    αr = unsafe_load(αr_)
    αi = unsafe_load(αi_)
    β  = unsafe_load(β_)
    return (αr^2 + αi^2 < (1 + SCHUR_EPSILON)*β^2)
end
    
function udomselect(αr_::Ptr, αi_::Ptr, β_::Ptr)::Cint
    αr = unsafe_load(αr_)
    αi = unsafe_load(αi_)
    β  = unsafe_load(β_)
    return (αr^2 + αi^2 >= (1 - SCHUR_EPSILON)*β^2)
end
    
function udopselect(αr_::Ptr, αi_::Ptr, β_::Ptr)::Cint
    αr = unsafe_load(αr_)
    αi = unsafe_load(αi_)
    β  = unsafe_load(β_)
    return (αr^2 + αi^2 >= (1 + SCHUR_EPSILON)*β^2)
end
    
function selectorder2(so::SCHURORDER)
    if so ==  lhpm
        return @cfunction(lhpmselect, Cint,
                          (Ptr{Cdouble}, Ptr{Cdouble}))
    elseif so ==  lhpp
        return @cfunction(lhppselect, Cint,
                          (Ptr{Cdouble}, Ptr{Cdouble}))
    elseif so ==  rhpm
        return @cfunction(rhpmselect, Cint,
                          (Ptr{Cdouble}, Ptr{Cdouble}))
    elseif so ==  rhpp
        return @cfunction(rhppselect, Cint,
                          (Ptr{Cdouble}, Ptr{Cdouble}))
    elseif so == udim
        return @cfunction(udimselect, Cint,
                          (Ptr{Cdouble}, Ptr{Cdouble}))
    elseif so ==  udip
        return @cfunction(udipselect, Cint,
                          (Ptr{Cdouble}, Ptr{Cdouble}))
    elseif so ==  udom
        return @cfunction(udomselect, Cint,
                          (Ptr{Cdouble}, Ptr{Cdouble}))
    elseif so ==  udop
        return @cfunction(udopselect, Cint,
                          (Ptr{Cdouble}, Ptr{Cdouble}))
        error("Unknown SCHURDORDER keyword: $so")
    end
end

function selectorder3(so::SCHURORDER)
    if so ==  lhpm
        return @cfunction(lhpmselect, Cint,
                          (Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}))
    elseif so ==  lhpp
        return @cfunction(lhppselect, Cint,
                          (Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}))
    elseif so ==  rhpm
        return @cfunction(rhpmselect, Cint,
                          (Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}))
    elseif so ==  rhpp
        return @cfunction(rhppselect, Cint,
                          (Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}))
    elseif so == udim
        return @cfunction(udimselect, Cint,
                          (Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}))
    elseif so ==  udip
        return @cfunction(udipselect, Cint,
                          (Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}))
    elseif so ==  udom
        return @cfunction(udomselect, Cint,
                          (Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}))
    elseif so ==  udop
        return @cfunction(udopselect, Cint,
                          (Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}))
        error("Unknown SCHURDORDER keyword: $so")
    end
end

function make_ggesselect(f::Function)
    function sfunc(αr, αi, β)::Cint
        return schurselect(f, αr, αi, β)
    end
    return sfunc
end

"""
    SchurWs

Workspace to be used with the [`LinearAlgebra.Schur`](https://docs.julialang.org/en/v1/stdlib/LinearAlgebra/#LinearAlgebra.GeneralizedSchur) representation
of the Schur decomposition which uses the [`LAPACK.gees!`](@ref) function.

# Examples
```jldoctest
julia> A = [1.2 2.3
            6.2 3.3]
2×2 Matrix{Float64}:
 1.2  2.3
 6.2  3.3

julia> ws = SchurWs(A)
SchurWs{Float64}
  work: 68-element Vector{Float64}
  wr: 2-element Vector{Float64}
  wi: 2-element Vector{Float64}
  vs: 2×2 Matrix{Float64}
  sdim: Base.RefValue{Int64}
  bwork: 2-element Vector{Int64}
  eigen_values: 2-element Vector{ComplexF64}

julia> t = Schur(LAPACK.gees!(ws, 'V', A)...)
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
mutable struct SchurWs{T<:AbstractFloat} <: Workspace
    work::Vector{T}
    wr::Vector{T}
    wi::Vector{T}
    vs::Matrix{T}
    sdim::Ref{BlasInt}
    bwork::Vector{BlasInt}
    eigen_values::Vector{Complex{T}}
end

Base.length(ws::SchurWs) = length(ws.wr)

for (gees, elty) in ((:dgees_, :Float64),
                     (:sgees_, :Float32))
    @eval begin
        function Base.resize!(ws::SchurWs, A::AbstractMatrix{$elty}; work=true)
            require_one_based_indexing(A)
            chkstride1(A)
            n     = checksquare(A)
            resize!(ws.wr, n)
            resize!(ws.wi, n)
            ws.vs = zeros($elty, n, n)
            resize!(ws.bwork, n)
            resize!(ws.eigen_values, n)
            if work
                info  = Ref{BlasInt}()
                ccall((@blasfunc($gees), liblapack), Cvoid,
                      (Ref{UInt8}, Ref{UInt8}, Ptr{Cvoid}, Ref{BlasInt},
                       Ptr{$elty}, Ref{BlasInt}, Ptr{Cvoid}, Ptr{$elty},
                       Ptr{$elty}, Ptr{$elty}, Ref{BlasInt}, Ptr{$elty},
                       Ref{BlasInt}, Ptr{Cvoid}, Ptr{BlasInt}, Clong, Clong),
                      'V', 'N', C_NULL, n,
                      A, max(1, stride(A, 2)), C_NULL, ws.wr,
                      ws.wi, ws.vs, max(size(ws.vs, 1), 1), ws.work,
                      -1, C_NULL, info, 1, 1)

                chklapackerror(info[])

                resize!(ws.work, BlasInt(real(ws.work[1])))
            end
            return ws
        end
            
        SchurWs(A::AbstractMatrix{$elty}) =
            resize!(SchurWs(Vector{$elty}(undef, 1), $elty[], $elty[], Matrix{$elty}(undef, 0, 0), Ref{BlasInt}(), BlasInt[], Complex{$elty}[]), A) 

        function gees!(ws::SchurWs{$elty}, jobvs::AbstractChar,
                       A::AbstractMatrix{$elty};
                       select::Union{Nothing,Function,SCHURORDER} = nothing,
                       resize=true)
            require_one_based_indexing(A)
            chkstride1(A)
            n = checksquare(A)
            nws = length(ws)
            if n != nws  
                if resize
                    resize!(ws, A; work = n > nws)
                else
                    throw(WorkspaceSizeError(nws, n))
                end
            end
            info = Ref{BlasInt}()
            ldvs = max(size(ws.vs, 1), 1)
            lwork = length(ws.work)
            if select isa Function
                sfunc = make_geesselect(select)
                sel_func = @cfunction($(Expr(:$, :sfunc)), Cint,
                                      (Ptr{Cdouble}, Ptr{Cdouble}))
                ccall((@blasfunc($gees), liblapack), Cvoid,
                      (Ref{UInt8}, Ref{UInt8}, Ptr{Cvoid}, Ref{BlasInt},
                       Ptr{$elty}, Ref{BlasInt}, Ptr{BlasInt}, Ptr{$elty},
                       Ptr{$elty}, Ptr{$elty}, Ref{BlasInt}, Ptr{$elty},
                       Ref{BlasInt}, Ptr{BlasInt}, Ptr{BlasInt}),
                      jobvs, 'S', sel_func, n,
                      A, max(1, stride(A, 2)), ws.sdim, ws.wr,
                      ws.wi, ws.vs, ldvs, ws.work,
                      lwork, ws.bwork, info)
            elseif select isa SCHURORDER
                sel_func = selectorder2(select)
                ccall((@blasfunc($gees), liblapack), Cvoid,
                      (Ref{UInt8}, Ref{UInt8}, Ptr{Cvoid}, Ref{BlasInt},
                       Ptr{$elty}, Ref{BlasInt}, Ptr{BlasInt}, Ptr{$elty},
                       Ptr{$elty}, Ptr{$elty}, Ref{BlasInt}, Ptr{$elty},
                       Ref{BlasInt}, Ptr{BlasInt}, Ptr{BlasInt}),
                      jobvs, 'S', sel_func, n,
                      A, max(1, stride(A, 2)), ws.sdim, ws.wr,
                      ws.wi, ws.vs, ldvs, ws.work,
                      lwork, ws.bwork, info)
            else
                ccall((@blasfunc($gees), liblapack), Cvoid,
                      (Ref{UInt8}, Ref{UInt8}, Ptr{Cvoid}, Ref{BlasInt},
                       Ptr{$elty}, Ref{BlasInt}, Ptr{BlasInt}, Ptr{$elty},
                       Ptr{$elty}, Ptr{$elty}, Ref{BlasInt}, Ptr{$elty},
                       Ref{BlasInt}, Ptr{Cvoid}, Ptr{BlasInt}),
                      jobvs, 'N', C_NULL, n,
                      A, max(1, stride(A, 2)), Ref{BlasInt}(), ws.wr,
                      ws.wi, ws.vs, ldvs, ws.work,
                      lwork, C_NULL, info)
            end

            chklapackerror(info[])

            if iszero(ws.wi)
                return A, ws.vs, ws.wr
            else
                @inbounds for i in axes(A, 1)
                    ws.eigen_values[i] = complex(ws.wr[i], ws.wi[i])
                end
                return A, ws.vs, ws.eigen_values
            end
        end
    end
end

"""
    gees!(ws, jobvs, A; select=nothing, resize=true) -> (A, vs, ws.eigen_values)

Computes the eigenvalues (`jobvs = N`) or the eigenvalues and Schur
vectors (`jobvs = V`) of matrix `A`, using the preallocated [`SchurWs`](@ref) worspace `ws`. If `ws` is not of the appropriate size and `resize==true` it will be resized for `A`. 
`A` is overwritten by its Schur form, and `ws.eigen_values` is overwritten with the eigenvalues.

It is possible to specify `select`, a function used to sort the eigenvalues during the decomponsition.
The function should have the signature `f(wr::T, wi::T) -> Bool`, where
`wr` and `wi` are the real and imaginary parts of the eigenvalue, and `T == eltype(A)`. 

Returns `A`, `vs` containing the Schur vectors, and `ws.eigen_values`.
"""
gees!(ws::SchurWs, jobvs::AbstractChar, A::AbstractMatrix; kwargs...)

"""
    GeneralizedSchurWs

Workspace to be used with the [`LinearAlgebra.GeneralizedSchur`](https://docs.julialang.org/en/v1/stdlib/LinearAlgebra/#LinearAlgebra.GeneralizedSchur)
representation of the Generalized Schur decomposition which uses the [`LAPACK.gges!`](@ref) function.

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

julia> ws = GeneralizedSchurWs(A)
GeneralizedSchurWs{Float64}
  work: 90-element Vector{Float64}
  αr: 2-element Vector{Float64}
  αi: 2-element Vector{Float64}
  β: 2-element Vector{Float64}
  vsl: 2×2 Matrix{Float64}
  vsr: 2×2 Matrix{Float64}
  sdim: Base.RefValue{Int64}
  bwork: 2-element Vector{Int64}
  eigen_values: 2-element Vector{ComplexF64}
  
julia> t = GeneralizedSchur(LAPACK.gges!(ws, 'V','V', A, B)...)
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
mutable struct GeneralizedSchurWs{T} <: Workspace
    work::Vector{T}
    αr::Vector{T}
    αi::Vector{T}
    β::Vector{T}
    vsl::Matrix{T}
    vsr::Matrix{T}
    sdim::Ref{BlasInt}
    bwork::Vector{BlasInt}
    eigen_values::Vector{Complex{T}}
end

Base.length(ws::GeneralizedSchurWs) = length(ws.αr)

# look into matlab function
for (gges, elty) in ((:dgges_, :Float64),
                     (:sgges_, :Float32))
    @eval begin
        function Base.resize!(ws::GeneralizedSchurWs, A::AbstractMatrix{$elty}; work=true)
            chkstride1(A)
            n     = checksquare(A)
            resize!(ws.αr, n)
            resize!(ws.αi, n)
            resize!(ws.β, n)
            resize!(ws.bwork, n)
            resize!(ws.eigen_values, n)
            ws.vsl = zeros($elty, n, n)
            ws.vsr = zeros($elty, n, n)
            if work
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
                      max(1, stride(A, 2)), C_NULL, ws.αr, ws.αi,
                      ws.β, ws.vsl, n, ws.vsr,
                      n, ws.work, -1, C_NULL,
                      info, 1, 1, 1)

                chklapackerror(info[])
                resize!(ws.work, BlasInt(real(ws.work[1])))
            end
            return ws
        end
        GeneralizedSchurWs(A::AbstractMatrix{$elty}) = 
            resize!(GeneralizedSchurWs(Vector{$elty}(undef, 1), $elty[], $elty[], $elty[], Matrix{$elty}(undef, 0, 0), Matrix{$elty}(undef, 0, 0), Ref{BlasInt}(), BlasInt[], Complex{$elty}[]), A)

        function gges!(ws::GeneralizedSchurWs, jobvsl::AbstractChar,
                       jobvsr::AbstractChar,
                       A::AbstractMatrix{$elty}, B::AbstractMatrix{$elty};
                       select::Union{Nothing,Function,SCHURORDER} = nothing,
                       resize=true)
            chkstride1(A, B)
            n = checksquare(A)
            m = checksquare(B)
            if n != m
                throw(DimensionMismatch("dimensions of A, ($n,$n), and B, ($m,$m), must match"))
            end
            nws = length(ws)
            if n != nws
                if resize
                    resize!(ws, A; work = n > nws)
                else
                    throw(WorkspaceSizeError(nws, n))
                end
            end

            info = Ref{BlasInt}()
            ldvsl = size(ws.vsl, 1)
            ldvsr = size(ws.vsr, 1)
            if select isa Function
                sfunc = make_ggesselect(select)
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
                      max(1, stride(B, 2)), ws.sdim, ws.αr, ws.αi,
                      ws.β, ws.vsl, ldvsl, ws.vsr,
                      ldvsr, ws.work, length(ws.work), ws.bwork,
                      info, 1, 1, 1)
            elseif select isa SCHURORDER
                sel_func = selectorder3(select)
                ccall((@blasfunc($gges), liblapack), Cvoid,
                      (Ref{UInt8}, Ref{UInt8}, Ref{UInt8}, Ptr{Cvoid},
                       Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt}, Ptr{$elty},
                       Ref{BlasInt}, Ref{BlasInt}, Ptr{$elty}, Ptr{$elty},
                       Ptr{$elty}, Ptr{$elty}, Ref{BlasInt}, Ptr{$elty},
                       Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt}, Ptr{Cvoid},
                       Ref{BlasInt}, Clong, Clong, Clong),
                      jobvsl, jobvsr, 'S', sel_func,
                      n, A, max(1, stride(A, 2)), B,
                      max(1, stride(B, 2)), ws.sdim, ws.αr, ws.αi,
                      ws.β, ws.vsl, ldvsl, ws.vsr,
                      ldvsr, ws.work, length(ws.work), ws.bwork,
                      info, 1, 1, 1)
            else
                ccall((@blasfunc($gges), liblapack), Cvoid,
                      (Ref{UInt8}, Ref{UInt8}, Ref{UInt8}, Ptr{Cvoid},
                       Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt}, Ptr{$elty},
                       Ref{BlasInt}, Ref{BlasInt}, Ptr{$elty}, Ptr{$elty},
                       Ptr{$elty}, Ptr{$elty}, Ref{BlasInt}, Ptr{$elty},
                       Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt}, Ptr{Cvoid},
                       Ref{BlasInt}, Clong, Clong, Clong),
                      jobvsl, jobvsr, 'N', C_NULL,
                      n, A, max(1, stride(A, 2)), B,
                      max(1, stride(B, 2)), Ref{BlasInt}(), ws.αr, ws.αi,
                      ws.β, ws.vsl, ldvsl, ws.vsr,
                      ldvsr, ws.work, length(ws.work), C_NULL,
                      info, 1, 1, 1)
            end
            chklapackerror(info[])
            @inbounds for i in axes(A, 1)
                ws.eigen_values[i] = complex(ws.αr[i], ws.αi[i])/ws.β[i]
            end
            return A, B, complex.(ws.αr, ws.αi), ws.β, ws.vsl, ws.vsr
        end
    end
end

"""
    gges!(ws, jobvsl, jobvsr, A, B; select=nothing, resize=true) -> (A, B, ws.α, ws.β, ws.vsl, ws.vsr)

Computes the generalized eigenvalues, generalized Schur form, left Schur
vectors (`jobsvl = V`), or right Schur vectors (`jobvsr = V`) of `A` and
`B`, using preallocated [`GeneralizedSchurWs`](@ref) workspace `ws`.
If `ws` is not of the right size, and `resize==true` it will be resized appropriately.

It is possible to specify `select`, a function used to sort the eigenvalues to the top left corner of the Schur form.
The function should have the signature `f(αr::T, αi::T, β::T) -> Bool` where `T == eltype(A)`. 
An eigenvalue `(αr[j]+αi[j])/β[j]` is selected if `f(αr[j],αi[j],β[j])` is true, 
i.e. if either one of a complex conjugate pair of eigenvalues is selected,
then both complex eigenvalues are selected.
The generalized eigenvalues components are returned in `ws.α` and `ws.β` where `ws.α` is a complex vector and `ẁs.β`, a real vector.
The generalized eigenvalues (`ws.α./ws.β`) are returned in `ws.eigen_values`, a complex vector. 
The left Schur vectors are returned in `ws.vsl` and the right Schur vectors are returned in `ws.vsr`.
"""
gges!(ws::GeneralizedSchurWs, jobvsl::AbstractChar, jobvsr::AbstractChar, A::AbstractMatrix,
      B::AbstractMatrix)
