struct SingularException <: Exception end

struct DggesException <: Exception
    error_nbr::Int64
end

Base.showerror(io::IO, e::DggesException) = print(io, "dgges error ", e.error_nbr)
