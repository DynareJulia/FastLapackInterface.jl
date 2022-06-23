using Documenter
using FastLapackInterface

DocMeta.setdocmeta!(FastLapackInterface, :DocTestSetup, :(using FastLapackInterface); recursive=true)

# Build documentation.
# ====================

makedocs(
    # options
    modules = [FastLapackInterface],
    # doctest = true,
    clean = false,
    sitename = "FastLapackInterface.jl",
    format=Documenter.HTML(; prettyurls=get(ENV, "CI", nothing) == "true"),
    pages = Any[
        "Introduction" => "index.md"
    ],
    strict = true
)

# Deploy built documentation from Travis.
# =======================================

deploydocs(
    # options
    repo = "github.com/DynareJulia/FastLapackInterface.jl.git",
    target = "build",
    deps = nothing,
    make = nothing,
    devbranch = "main"
)