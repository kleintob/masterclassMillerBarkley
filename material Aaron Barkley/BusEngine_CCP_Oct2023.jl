using StatsKit, ForwardDiff, Ipopt, NLsolve, Optim, Parameters, Zygote, LinearAlgebra, Random, Plots, BenchmarkTools
"""
Conditional choice probability estimation of the Rust (1987) bus engine replacement model
"""

"""
These two functions perform value function iteration and generate the data
"""

function value_function_iteration(X::AbstractRange{Float64},S::Vector{Int64},F1::Matrix{Float64},F2::Matrix{Float64},β::Number,θ::Vector;MaxIter=1000)
    x_len=length(X);
    γ=0.5772;
    value_function2=zeros(x_len,length(S));
    value_diff=1.0;
    tol=1e-5;
    iter=1;
    local v1, v2
    while (value_diff>tol) && (iter<=MaxIter)
        value_function1=value_function2;
        v1=[0.0 + β*F1[j,:]'*value_function1[:,s] for j∈eachindex(X), s∈eachindex(S)];
        v2=[θ[1]+θ[2]*X[j]+θ[3]*S[s] + β*(F2[j,:]'*value_function1[:,s]) for j=1:x_len, s∈eachindex(S)];
        value_function2=[log(exp(v1[j,s])+exp(v2[j,s]))+γ for j=1:x_len, s=1:length(S)];
        iter=iter+1;
        #value_diff=sum((value_function1 .- value_function2).^2);
        value_diff=maximum((value_function1 .- value_function2).^2);
    end
    ccps=[1/(1+exp(v2[j,s,]-v1[j,s])) for j=1:x_len, s=1:length(S)];
    return (ccps_true=ccps, value_function=value_function2)
end

function generate_data(N,T,X,S,F1,F2,F_cumul,β,θ;T_init=10,π=0.4,ex_initial=0)
    if ex_initial==1
        T_init=0;
    end
    x_data=zeros(N,T+T_init);
    x_data_index=Array{Int32}(ones(N,T+T_init));
    if ex_initial==1
        x_data_index[:,1]=rand(1:length(X),N,1);
        x_data[:,1]=X[x_data_index[:,1]];
    end
    s_data=(rand(N) .> π) .+ 1;
    d_data=zeros(N,T+T_init);

    draw_ccp=rand(N,T+T_init);
    draw_x=rand(N,T+T_init);

    (ccps,_)=value_function_iteration(X,S,F1,F2,β,θ);

    for n=1:N
        for t=1:T+T_init
            d_data[n,t]=(draw_ccp[n,t] > ccps[x_data_index[n,t],s_data[n]])+1;
            if t<T+T_init
                x_data_index[n,t+1]=1 + (d_data[n,t]==2)*sum(draw_x[n,t] .> F_cumul[x_data_index[n,t],:]); 
                x_data[n,t+1]=X[x_data_index[n,t+1]];
            end
        end
    end

    return (XData=x_data[:,T_init+1:T+T_init], SData=repeat(s_data,1,T),
        DData=d_data[:,T_init+1:T+T_init],
        XIndexData=x_data_index[:,T_init+1:T_init+T],
        TData=repeat(1:T,N,1),
        NData=repeat((1:N)',1,T)) 
end

"""
Stationary infinite horizon estimation
"""
function inf_horizon_ccp_estimation(DData, XData, SData, X, S, F1, F2, β)
    x_len = length(X);
    b=ccp_est_logit(DData[:],XData[:],SData[:]);
    All_States = [X 1.0*ones(x_len,1); X 2.0*ones(x_len,1)];
    W_fit=[ones(size(All_States,1),) All_States[:,1] All_States[:,1].^2 (All_States[:,2].==2.0) (All_States[:,2].==2.0).*All_States[:,1]];
    ccp_hat=reshape(exp.(W_fit*b)./(1 .+ exp.(W_fit*b)),x_len,length(S));

    All_States_Index=[findfirst(all(All_States .== [XData[j] SData[j]], dims=2)[:,1]) for j∈eachindex(XData)];

    F2_all=[F2 zeros(x_len,x_len); zeros(x_len,x_len) F2];
    F1_all=[F1 zeros(x_len,x_len); zeros(x_len,x_len) F1];

    γ = 0.5772;
    V1 = inv(I(size(All_States,1)) .- β*(repeat(ccp_hat[:],1,size(All_States,1)).*F1_all .+ 
            repeat(1.0 .- ccp_hat[:],1,size(All_States,1)).*F2_all ));
    V2 = ccp_hat[:].*(γ .- log.(ccp_hat[:])) .+ (1.0 .- ccp_hat[:]).*(γ .- log.(1.0 .- ccp_hat[:]));
    V3 = (1.0 .- ccp_hat[:]).*[ones(size(All_States,1),1) All_States];

    f(θ) = stationary_lik(θ,DData,V1,V2,V3,F1_all,F2_all,β,All_States,All_States_Index)
    result = optimize(f, [1.0;1.0;1.0], LBFGS(); autodiff = :forward);
    return result.minimizer

end

function stationary_lik(θ, DData,V1,V2,V3,F1_tot,F2_tot,β,All_States,All_States_Index)

    V_exante = V1*(V2 + V3*θ);
    v_diff = [ones(size(All_States_Index,1),1) All_States[All_States_Index,:]]*θ .+ 
            β*(F2_tot[All_States_Index,:] .- F1_tot[All_States_Index,:])*V_exante;
    return -sum((DData[:] .==2).*v_diff .- log.(1 .+ exp.(v_diff))) 
end

"""
CCP estimation exploiting renewal action
"""
function ccp_est_logit(D_data,X_data,S_data)
    W_ccp=[ones(size(X_data,1),) X_data X_data.^2 (S_data.==2.0) (S_data.==2.0).*X_data];
    logit_lik(b) = -sum((D_data.==1.0).*(W_ccp*b) .- log.(1 .+ exp.(W_ccp*b)))
    x_0= 0.1.*ones(size(W_ccp,2),1);
    result=optimize(logit_lik,x_0, LBFGS(),Optim.Options(g_tol = 1e-6); autodiff = :forward)
    return result.minimizer
end

function ccp_estimation(DData,XData,SData,X,S,F1,F2,β)
    x_len=length(X);
    b=ccp_est_logit(DData[:],XData[:],SData[:]);
    All_States = [X 1.0*ones(x_len,1); X 2.0*ones(x_len,1)];
    W_fit=[ones(size(All_States,1),) All_States[:,1] All_States[:,1].^2 (All_States[:,2].==2.0) (All_States[:,2].==2.0).*All_States[:,1]];
    ccp_hat=reshape(exp.(W_fit*b)./(1 .+ exp.(W_fit*b)),x_len,length(S));

    All_States_Index=[findfirst(all(All_States .== [XData[j] SData[j]], dims=2)[:,1]) for j∈eachindex(XData)];

    γ=0.5772;
    F2_all=[F2 zeros(x_len,x_len); zeros(x_len,x_len) F2];
    F1_all=[F1 zeros(x_len,x_len); zeros(x_len,x_len) F1];
    fv = β*(F1_all[All_States_Index,:] .- F2_all[All_States_Index,:])*(γ .- log.(ccp_hat[:]));

    ccp_lik(b) = -sum((DData[:].==2.0).*([ones(size(XData[:],1),) All_States[All_States_Index,:]]*b .- fv) .- 
                log.(1 .+ exp.([ones(size(XData[:],1),) All_States[All_States_Index,:]]*b .- fv)));
    result=optimize(ccp_lik,[0.1;0.1;0.1], LBFGS(); autodiff = :forward);
    return result.minimizer
end

"""
Full information maximum likelihood estimation
"""
function fiml_estimation(DData::Matrix,XData::Matrix,SData::Matrix{Int64},X,S,F1,F2,β)
    x_len = length(X);
    All_States = [X 1.0*ones(x_len,1); X 2.0*ones(x_len,1)];
    All_States_Index=[findfirst(all(All_States .== [XData[j] SData[j]], dims=2)[:,1]) for j∈eachindex(XData)];
    f(θ) = fiml_likelihood(θ, DData,All_States_Index,X,S,F1,F2,β)
    result = optimize(f,[0.1;0.1;0.1], LBFGS(),Optim.Options(g_tol = 1e-6); autodiff = :forward);
    #result = optimize(f,[0.1;0.1;0.1]);
    return result.minimizer
end


function fiml_likelihood(θ,DData,All_States_Index,X,S,F1,F2,β)
    _,value_function = value_function_iteration(X,S,F1,F2,β,θ)
    v1=[0.0 + β*F1[j,:]'*value_function[:,s] for j∈eachindex(X), s∈eachindex(S)];
    v2=[θ[1]+θ[2]*X[j]+θ[3]*S[s] + β*(F2[j,:]'*value_function[:,s]) for j∈eachindex(X), s∈eachindex(S)];

    return -sum((DData[j] == 2.0) * (v2[All_States_Index[j]] - v1[All_States_Index[j]] ) - log(1 + exp(v2[All_States_Index[j]]-v1[All_States_Index[j]])) for j∈eachindex(All_States_Index))

end

"""
Simulate data and estimate each model above
"""
function main()

    x_min=0.0;
    x_max=15.0;
    x_int=0.05;
    x_len=Int32(1+(x_max-x_min)/x_int);
    x=range(x_min,x_max,x_len);

    # Transition matrix for mileage:
    x_tran       = zeros((x_len, x_len));
    x_tran_cumul = zeros((x_len, x_len));
    x_tday      = repeat(x, 1, x_len); 
    x_next      = x_tday';
    x_zero      = zeros((x_len, x_len));

    x_tran = (x_next.>=x_tday) .* exp.(-(x_next - x_tday)) .* (1 .- exp(-x_int));
    x_tran[:,end]=1 .-sum(x_tran[:,1:(end-1)],dims=2);
    x_tran_cumul=cumsum(x_tran,dims=2);

    S=[1, 2];
    s_len=Int32(length(S));
    F1=zeros(x_len,x_len);
    F1[:,1].=1.0;
    F2=x_tran;

    N=1000;
    T=15;
    X=x;
    θ=[2.0, -0.15, 1.0];
    β=0.9;
    F_cumul=x_tran_cumul;
    Random.seed!(2023);
    XData, SData, DData, XIndexData, TData, NData = generate_data(N,T,X,S,F1,F2,F_cumul,β,θ);

    theta_hat_ccp = ccp_estimation(DData,XData,SData,X,S,F1,F2,β);
    theta_hat_fiml = fiml_estimation(DData,XData,SData,X,S,F1,F2,β);
    inf_horz_ccp = inf_horizon_ccp_estimation(DData, XData, SData, X, S, F1, F2, β);
    return theta_hat_ccp, theta_hat_fiml, inf_horz_ccp, θ
end

"""
Collect results and display
"""
function display_results()
    theta_hat_ccp, theta_hat_fiml, inf_horz_ccp, θ = main();

    θ_1, θ_2, θ_3 = round.(θ, digits = 4)

    theta_hat_1, theta_hat_2, theta_hat_3 = round.(theta_hat_ccp, digits = 4)
    println("----------------------------")
    println("Finite Dependence CCP: Estimation results \n ----------------------------")
    println("True θ_1: $θ_1 \nEstimated θ_1: $theta_hat_1\n")
    println("True θ_2: $θ_2 \nEstimated θ_2: $theta_hat_2 \n")
    println("True θ_3: $θ_3 \nEstimated θ_3: $theta_hat_3 \n")


    theta_hat_1, theta_hat_2, theta_hat_3 = round.(inf_horz_ccp, digits = 4)
    println("----------------------------")
    println("Stationary Infinite Horizon CCP: Estimation results \n ----------------------------")
    println("True θ_1: $θ_1 \nEstimated θ_1: $theta_hat_1\n")
    println("True θ_2: $θ_2 \nEstimated θ_2: $theta_hat_2 \n")
    println("True θ_3: $θ_3 \nEstimated θ_3: $theta_hat_3 \n")


    theta_hat_1, theta_hat_2, theta_hat_3 = round.(theta_hat_fiml, digits = 4)
    println("----------------------------")
    println("Stationary Infinite Horizon CCP: Estimation results \n ----------------------------")
    println("True θ_1: $θ_1 \nEstimated θ_1: $theta_hat_1\n")
    println("True θ_2: $θ_2 \nEstimated θ_2: $theta_hat_2 \n")
    println("True θ_3: $θ_3 \nEstimated θ_3: $theta_hat_3 \n")

end

display_results()