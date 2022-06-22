# Based on documentation of DFControl and DFTK
using LibGit2: LibGit2
using Pkg: Pkg

# To manually generate the docs:
#
# 1. Install all python dependencies from the PYDEPS array below.
# 2. Run "julia make.jl" to generate the docs

# Set to true to disable some checks and cleanup
DEBUG = false

# Where to get files from and where to build them
SRCPATH = joinpath(@__DIR__, "src")
BUILDPATH = joinpath(@__DIR__, "build")
ROOTPATH = joinpath(@__DIR__, "..")
CONTINUOUS_INTEGRATION = get(ENV, "CI", nothing) == "true"

# Python and Julia dependencies needed for running the notebooks
# PYDEPS = ["ase", "pymatgen"]
JLDEPS = [Pkg.PackageSpec(;
                          url = "https://github.com/louisponet/FastLapackInterface.jl.git",
                          rev = LibGit2.head(ROOTPATH))]

# Setup julia dependencies for docs generation if not yet done
Pkg.activate(@__DIR__)
# if !isfile(joinpath(@__DIR__, "Manifest.toml"))
Pkg.develop(Pkg.PackageSpec(; path = ROOTPATH))
Pkg.instantiate()
Pkg.add(; url = "https://github.com/kimikage/Documenter.jl", rev = "ansicolor")
# end

# Setup environment for making plots
ENV["GKS_ENCODING"] = "utf8"
ENV["GKSwstype"] = "100"
ENV["PLOTS_TEST"] = "true"

# Import packages for docs generation
using FastLapackInterface
using LinearAlgebra
using LinearAlgebra: LAPACK
using Documenter
using Literate

DocMeta.setdocmeta!(FastLapackInterface, :DocTestSetup,
                    quote
                        using FastLapackInterface, LinearAlgebra
                    end)

# Collect examples from the example index (src/index.md)
# The chosen examples are taken from the examples/ folder to be processed by Literate
EXAMPLES = [String(m[1])
            for m in
                match.(r"\"(examples/[^\"]+.md)\"",
                       readlines(joinpath(SRCPATH, "index.md"))) if !isnothing(m)]

# Collect files to treat with Literate (i.e. the examples and the .jl files in the docs)
# The examples go to docs/literate_build/examples, the .jl files stay where they are
literate_files = NamedTuple{(:src, :dest, :example),Tuple{String,String,Bool}}[(src = joinpath(ROOTPATH,
                                                                                               splitext(file)[1] *
                                                                                               ".jl"),
                                                                                dest = joinpath(SRCPATH,
                                                                                                "examples"),
                                                                                example = true)
                                                                               for file in
                                                                                   EXAMPLES]
for (dir, directories, files) in walkdir(SRCPATH)
    for file in files
        if endswith(file, ".jl")
            push!(literate_files, (src = joinpath(dir, file), dest = dir, example = false))
        end
    end
end

# Run Literate on them all
for file in literate_files
    # preprocess = file.example ? add_badges : identity
    # Literate.markdown(file.src, file.dest; documenter=true, credit=false,
    #                   preprocess=preprocess)
    # Literate.notebook(file.src, file.dest; credit=false,
    #                   execute=CONTINUOUS_INTEGRATION || DEBUG)
    Literate.markdown(file.src, file.dest; documenter = true, credit = false)
    Literate.notebook(file.src, file.dest; credit = false,
                      execute = CONTINUOUS_INTEGRATION || DEBUG)
end

# Generate the docs in BUILDPATH
makedocs(; modules = [FastLapackInterface],
         format = Documenter.HTML(
                                  # Use clean URLs, unless built as a "local" build
                                  ; prettyurls = CONTINUOUS_INTEGRATION,
                                  canonical = "https://dynarejulia.github.io/FastLapackInterface.jl/stable/",
                                  assets = ["assets/favicon.ico"]),
         sitename = "FastLapackInterface.jl", authors = "Louis Ponet, Michel Juillard",
         linkcheck = false,  # TODO
         linkcheck_ignore = [
                             # Ignore links that point to GitHub's edit pages, as they redirect to the
                             # login screen and cause a warning:
                             r"https://github.com/([A-Za-z0-9_.-]+)/([A-Za-z0-9_.-]+)/edit(.*)"],
         pages = ["Home" => "index.md",
                  "Work Spaces" => "workspaces.md",
                  "LAPACK" => "LAPACK.md"
                  # "Getting started" => Any["guide/installation.md",
                  #                          "guide/configuration.md",
                  #                          "Basic Tutorial"=>"guide/basic_tutorial.md",
                  #                          "Advanced Tutorial"=>"guide/advanced_tutorial.md"],
                  # "Usage" => Any["guide/jobs.md", "guide/calculations.md", "guide/servers.md",
                  #                "guide/structure.md",],
                  # "Examples" => EXAMPLES,
                  # "api.md"
                  # "publications.md",
                  ]
         # strict = !DEBUG,
         )

# Dump files for managing dependencies in binder
if CONTINUOUS_INTEGRATION
    cd(BUILDPATH) do
        open("environment.yml", "w") do io
            return print(io, """
                             name: fastlapackinterface
                             """)
        end

        # Install Julia dependencies into build
        Pkg.activate(".")
        for dep in JLDEPS
            Pkg.add(dep)
        end
    end
    Pkg.activate(@__DIR__)  # Back to Literate / Documenter environment
end

# Deploy docs to gh-pages branch
deploydocs(; repo = "github.com/DynareJulia/FastLapackInterface.jl.git", devbranch = "main")

# Remove generated example files
if !DEBUG
    for file in literate_files
        base = splitext(basename(file.src))[1]
        for ext in [".ipynb", ".md"]
            rm(joinpath(file.dest, base * ext); force = true)
        end
    end
end

if !CONTINUOUS_INTEGRATION
    println("\nDocs generated, try $(joinpath(BUILDPATH, "index.html"))")
end
