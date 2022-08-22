struct SingularException <: Exception end

struct DggesException <: Exception
    error_nbr::Int
end

Base.showerror(io::IO, e::DggesException) = print(io, "dgges error ", e.error_nbr)

struct WorkspaceSizeError <: Exception
    nws::Int
    n::Int
end
Base.showerror(io::IO, e::WorkspaceSizeError) = print(io, "Workspace has the wrong size: expected $(e.n), got $(e.nws).\nUse resize!(ws, A).")
