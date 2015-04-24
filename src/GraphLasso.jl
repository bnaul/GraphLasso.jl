module GraphLasso
using DataFrames, Lasso, Graphs

export graphlasso

function graphlasso(S::Matrix{Float64}, α::Float64; tol::Float64=1e-5,
    maxit::Int=1000, penalize_diag::Bool=true)
    p = size(S,1)
    adj = abs(S) .> α
    blocks = connected_components(sparse2adjacencylist(sparse(adj*1.0)))
    print(blocks)
    W = zeros(p,p)
    W_old = copy(W)
    Θ = zeros(p,p)
    for k = 1:length(blocks)
        W[blocks[k],blocks[k]], Θ[blocks[k],blocks[k]] = fit_block(S[blocks[k],blocks[k]], α, tol, maxit, penalize_diag)
    end
    return W, Θ 
end

function fit_block(S::Matrix{Float64}, α::Float64, tol::Float64, maxit::Int, penalize_diag::Bool)
    p = size(S,1)
    W = copy(S)
    if penalize_diag
        W += α * eye(p)
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
            sqrtW11 = sqrtm(W11)
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
end
