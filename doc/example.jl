#! /bin/julia
using SequentialFit, Plots
N = 20 # Number of frames

gaussian(x,A,mu,sigma) = A*exp(-((x-mu)/sigma)^2)
p_true = [0.7,-1.5,0.4,0.9,1.2,0.7]

function slowFunction(x)
    # sleep(2) # slow evaluation, disabled for illustration
    +(
        gaussian(x,p_true[1:3]...), # Gaussian peak
        gaussian(x,p_true[4:6]...), # Gaussian peak
        0.05sin(15*x) # noise
       )
end

@. model(x,p) = gaussian(x,p[1:3]...) + gaussian(x,p[4:6]...)
p_0 = [1.0,-1.0,0.5,1.0,1.0,0.5] # Initial guess of parameters. Has to be decent
x_0 = [-2.0, -1.0, 0.0, 1.0] # Initial sample points

s = SeqFit(slowFunction, model, x_0, p_0)
err = zeros(N,length(p_0))

anim = @animate for i in 1:N
    update!(s)
    err[i,:] .= abs.(s.fit.param ./ p_true .- 1)
    plot(
         s,groundtruth=true,
         xlim=(-5,5),ylim=(0,1),
        )
end

gif(anim, "doc/example.gif",fps=1)

savefig(
        plot(
             err,
             legend=false,
             xlabel="Iteration",
             ylabel="Error",
            ),
        "doc/example_errors.png",
       )
