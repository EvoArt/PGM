cd("C:\\Users\\arn203\\OneDrive - University of Exeter\\Documents\\PGM")
] activate .
using LinearAlgebra,Calculus, MCMCChains, StatsPlots, Distributions, BenchmarkTools,
Tullio, LoopVectorization,StatsFuns, StatsBase



gα(θα,fαXαm, N,E,σ₀²) =fαXαm -N*E - θα/σ₀²
g(θ,fαXα,N,E,σ₀²) = gα.(θ,fαXα,N,E,σ₀²)
h(a,Varfα,gα,σ₀²,N) = exp((-a^2 /2)*(1/σ₀² + Varfα)+ a*N*gα)
Q(prob,proposal,old_val) = rand() < prob ? proposal : old_val
conv(b,c) = (b^-1)*c,b^(-1/2)

function get_q(Varfα ::Float64,gα ::Float64,σ₀² ::Float64,N ::Int64, Δ ::Float64)
    b = 1/σ₀² + Varfα
    c =  gα*N
    return truncated(Normal(conv(b,c)...),-Δ,Δ)
end

function get_q(Varfα ::Float64,gα ::Float64,σ₀² ::Float64,N ::Int64)
    b = 1/σ₀² + Varfα
    c =  gα*N
    return Normal(conv(b,c)...)
end

function unpack_PGM(A,n,xs,s)
    θa = [A[i,i] for i in 1:n]
    θb = [A[i,j] for i in 1:n for j in 1:i-1]
    return vcat(θa,θb)
end

function get_expectations(xs ::BitMatrix,s ::Int64,n ::Int64)
    Ea = sum(xs,dims = 1 )' ./s  
    Eb = [sum(prod(xs[:,[i,j]], dims = 2)) for i in 1:n for j in 1:i-1] ./s
return vec(vcat(Ea,Eb))
end

function get_var(xs ::BitMatrix,n ::Int64)
    Va = xs  
    Vb = hcat([prod(xs[:,[i,j]], dims = 2) for i in 1:n for j in 1:i-1]...)
return var(hcat( Va,Vb),dims = 1)'#sqrt(cov(hcat( Ea,Eb)))
end

function get_vars(xs ::BitMatrix,n ::Int64)
    Va = var(xs,dims = 1 )'
    Vb = [var(prod(xs[:,[i,j]], dims = 2)) for i in 1:n for j in 1:i-1] 
return vec(vcat(Va,Vb))
end

function get_names(name_list ::Vector{String},n ::Int64)
    na = name_list[1:n]   
    nb = [name_list[i] *"_"*name_list[j] for i in 1:n for j in 1:i-1] 
return vcat("p0","σ₀²",na...,nb...)
end


function get_indeces(n ::Int64)
    gibs_inds = [vcat([(i,j) for j in 1:i]...,[(j,i) for j in i+1:n]) for i in 1:n]
  θ_inds = [insert!([((pair[1])* ((n))  ) + (pair[2])  for pair in gibs_inds[i]][1:n .!==i],i,i) for i in 1:n]
    return Tuple([denserank(vcat(θ_inds'...))[i,:] for i in 1:n])
end

function gibbs_step(θ ::Vector{Float64},x ::BitVector,i ::Int64,inds ::Vector{Int64})
    x[i] = 1
    pos_set = inds[x .>0]
     logistic(sum(view(θ,pos_set))) < 1 || println(view(θ,pos_set))
     logistic(sum(view(θ,pos_set))) > 0 || println(view(θ,pos_set))

    x[i] = rand(Bernoulli(logistic(sum(view(θ,pos_set)))))
    return x
end

function gibs_set(θ ::Vector{Float64},x ::BitVector, n ::Int64,inds ::Tuple)
    for i in sample(1:n,n, replace = false)
        x = gibbs_step(θ,x,i,inds[i])
    end
    return x
end

function sampler(X,nSteps,ϵ,σ₀²,nGibbs,n,N;a = 1, b = 5,c = 5, d = 5)
    fαXα = get_expectations(X,1,n)
    # Setup
    inds = get_indeces(n)
    len_θ = Int(n + (n * (n-1))/2)
    edge_mask = 1:len_θ .> n
    p0 = rand(Beta(a, b))
    σ₀² = 1/ rand(Gamma(c, d))^2 
    θ = rand(Normal(0,σ₀²),len_θ)
    pt = randn(len_θ)
    xs = BitMatrix(undef,nGibbs,n)
    burnxs = Array{Float64}(undef,nSteps ÷4,len_θ)
    chain = Array{Float64}(undef, nSteps,len_θ +2)
    #gibbs
    x = Bool.(zeros(n))
    C = fill(1.0,len_θ)
    # Run chains
    for step in 1:nSteps
        sig = vcat(fill(2,n)...,fill(σ₀²,len_θ-n)...)
        chain[step,:] = vcat(p0,σ₀²,θ) #update chain
        #Gibbs samples and compute expectations
       # println(θ)
        for i in 1:nGibbs
            x = gibs_set(θ,x, n,inds)
            xs[i,:] = x
        end
        if step <= nSteps ÷4

           burnxs[step,:] = (get_var(xs,n) .+ 1 ./sig)'
           
        end
        if step == nSteps ÷4
         #  C = vec(sqrt.(mean(burnxs,dims = 1)))
           println(vec(sqrt.(mean(burnxs,dims = 1))))
        end
        E = get_expectations(xs,nGibbs,n)
        Varf = get_vars(xs,n)
        # Langevin dynamics
        lav_mask = θ .!= 0       
        pt,θ[lav_mask] = langevin(pt,ϵ,θ,fαXα,N,E,sig,lav_mask,  C)
        #pt,θ[lav_mask] = langevin(pt,ϵ,θ,fαXα,N,E,σ₀²,lav_mask, C = C)
        # Metropolis-Hastings
        θ = MH(θ,Varf,fαXα,N,E,σ₀²,edge_mask,p0)  
        # Hyper parameters
        p0 =rand(Beta(a +sum(θ[edge_mask] .>0), b +sum(θ[edge_mask] .==0)))
        σ₀² = 1/ rand(Gamma(c + 0.5*sum(θ[edge_mask] .>0), d + 0.5*0.5*sum((θ[edge_mask] .>0) .* (θ[edge_mask] .^2))))
    end
    return chain
end

function MH(θ ::Vector{Float64},Varfα ::Vector{Float64},fαXα ::Vector{Float64},N ::Int64,E ::Vector{Float64},σ₀² ::Float64,edge_mask ::BitVector,p0 ::Float64,Padd ::Float64 = 0.5, Pdel ::Float64 = 0.5)
    Δ = 0.01 ./sqrt.(Varfα*N)
    add_mask = (θ .== 0) .* edge_mask  .* (rand(length(θ)) .< Padd)
    del_mask = (θ .!= 0) .& (abs.(θ) .< Δ)  .* edge_mask .* (rand(length(θ)) .< Pdel)
    #add
    if sum(add_mask) >0
        gα = vec(g.(θ[add_mask],fαXα[add_mask],N,E[add_mask],σ₀²))
        add!(view(θ, add_mask ),vec(view(Varfα, add_mask )),gα,σ₀²,N,p0)
    end
    #delete
    if sum(del_mask) >0
        gα = g.(θ[del_mask],fαXα[del_mask],N,E[del_mask] ,σ₀²)
    #E[del_mask] .- (θ[del_mask] .* Varfα[del_mask]),σ₀²)
        del!(view(θ, del_mask ),view(Varfα, del_mask ),gα,σ₀²,N,p0)
    end
    return θ
end

function add!(A0 ::SubArray,Varfα ::SubArray,gα ::Vector{Float64},σ₀² ::Float64,N ::Int64,p0 ::Float64 = 0.5 ) 
    #proposals = rand.(q,length(A0))
    Δ = 0.01 ./sqrt.(Varfα*N)
    q = get_q.(Varfα,gα,σ₀²,N, Δ)
    proposals = rand.(q)
    q = get_q.(Varfα,gα,σ₀²,N)
    probs = (p0/(1-p0)) .* h.(proposals,Varfα,gα,σ₀²,N) ./pdf.(q,proposals)
    A0 .= Q.(probs,proposals,0);
end

function del!(A1 ::SubArray,Varfα ::SubArray,gα ::Vector{Float64},σ₀² ::Float64,N ::Int64, p0 ::Float64 = 0.5)
    
    proposals = zeros(length(A1))
    
    #probs =((1-p0)/p0) .* pdf(q,A1) ./ h.(A1,Varfα,gα,σ₀²,N)
    Δ = 0.01 ./sqrt.(Varfα*N)
    probs = 1 ./((p0/(1-p0)) .* h.(A1,Varfα,gα,σ₀²,N) ./ pdf.(get_q.(Varfα,gα,σ₀²,N),A1) )

    A1 .= Q.(probs,proposals,A1)

end

function langevin(pₜ ::Vector{Float64},ϵ ::Float64,θₜ ::Vector{Float64},fαXα ::Vector{Float64},N ::Int64,E ::Vector{Float64},σ₀² ::Vector{Float64},lav_mask ::BitVector, C ::Vector{Float64}, α  ::Float64 = sqrt(0.5), β ::Float64 = sqrt(0.5) )
    nₜ = randn(length(θₜ))
    pₜ = α .*pₜ .+ β*nₜ
    ptε2 =pₜ .+ ((ϵ*C/2) .* g.(θₜ,fαXα,N,E,σ₀²))
    θₜϵ = θₜ .+ (ϵ*C .*ptε2)
    pₜε = ptε2 .+ ((ϵ*C/2) .* g.(θₜϵ,fαXα,N,E,σ₀²))
    return  pₜε, θₜϵ[lav_mask]
end

X = Bool.(hcat(zeros(10),ones(10),ones(10)))#[1,1,1,1,1,0,0,0,0,0]))
samps = sampler(X,100000,0.01,0.2,30,3,10)#
plotly()
plot(Chains(samps))


JuliaSysimage.dll
