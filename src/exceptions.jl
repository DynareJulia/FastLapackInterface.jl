struct SingularException <: Exception end

struct DggesException <: Exception
    error_nbr::Int64
end

Base.showerror(io::IO, e::DggesException) = print(io, "error nbr", e.error_nbr)
