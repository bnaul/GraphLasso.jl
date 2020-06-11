module GraphLasso
using LinearAlgebra, SparseArrays
using DataFrames, Lasso, Graphs

export graphlasso

function graphlasso(S::Matrix{Float64}, α::Float64; tol::Float64=1e-5,
                    maxit::Int=1000, penalize_diag::Bool=true, verbose::Bool=false)
    p = size(S,1)
    adj = convert(SparseMatrixCSC{Int64,Int64}, abs.(S) .> α)
    blocks = connected_components(sparse2adjacencylist2(max.(adj, sparse(I, p, p))))
    if(verbose)
        print(blocks)
    end
    W = zeros(p,p)
    Θ = zeros(p,p)
    for block = blocks
        W[block,block], Θ[block,block] = fit_block(S[block,block], α, tol, maxit, penalize_diag)
    end
    return W, Θ
end

function fit_block(S::Matrix{Float64}, α::Float64, tol::Float64,
                   maxit::Int, penalize_diag::Bool)
    p = size(S,1)
    W = copy(S)
    if penalize_diag
        W += α*I
    end
    if size(S,1) == 1
        return W, inv(W)
    end
    Θ = zeros(p,p)
    i = 0
    β = zeros(p-1,p)
    while i < maxit
        i += 1
        W_old = copy(W)
        for j = 1:p
            inds = collect(1:p)
            splice!(inds, j)
            W11 = W[inds,inds]
            sqrtW11 = sqrt(W11)
            β[:,j] = fit(LassoPath, sqrtW11, sqrtW11 \ S[inds,j], λ=[α/(p-1)],
                         standardize=false, intercept=false).coefs
            W[inds,j] = W11 * β[:,j]
            W[j,inds] = W[inds,j]
        end
        if norm(W - W_old) < tol
            break
        end
    end
    if i == maxit
        warn("Maximum number of iterations reached, graphlasso failed to converge")
    end
    for j in 1:p
        inds = collect(1:p)
        splice!(inds, j)
        Θ[j,j] = 1/(W[j,j] - dot(W[inds,j],β[:,j]))
        Θ[inds,j] = -Θ[j,j] * β[:,j]
        Θ[j,inds] = Θ[inds,j]
    end
    return W, Θ
end

# Modify code from unmaintained Graph.jl
function sparse2adjacencylist2(A::SparseMatrixCSC{Tv,Ti}) where {Tv,Ti<:Integer}
    colptr = A.colptr
    rowval = A.rowval
    n = size(A, 1)
    adjlist = Array{Array{Ti,1}}(undef, n)
    s = 0
    for j in 1:n
        adjj = Ti[]
        sizehint!(adjj, colptr[j+1] - colptr[j] - 1)
        for k in colptr[j]:(colptr[j+1] - 1)
            rvk = A.rowval[k]
            if rvk != j push!(adjj, rvk) end
        end
        s += length(adjj)
        adjlist[j] = adjj
    end
    GenericAdjacencyList{Ti, UnitRange{Ti}, Vector{Vector{Ti}}}(!ishermitian(A),
                                                             one(Ti):convert(Ti,n),
                                                             s, adjlist)
end



end

