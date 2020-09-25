using CCBlade

import ForwardDiff, ReverseDiff
import SparseArrays, SparseDiffTools, FiniteDiff

import Statistics: median
using BenchmarkTools


# ------- setup problem for a number of flight states (run once) --------

function setup(nV)

    chord = 0.10
    D = 1.6
    RPM = 2100
    pitch = 1.0  # pitch distance in meters.

    turbine = false
    Rhub = 0.01
    Rtip = D/2
    B = 2  # number of blades

    R = D/2.0
    n = 11
    r = range(R/10, stop=9/10*R, length=n)
    theta = atan.(pitch./(2*pi*r))
    chord = chord*ones(n)

    function affunc(alpha, Re, M)

        cl = 6.2*alpha
        cd = 0.008 - 0.003*cl + 0.01*cl*cl

        return cl, cd
    end 

    pitch = 0.0
    precone = 0.0

    if nV == 1
        Vinf = [30.0]
    else
        Vinf = range(29.0, 31.0, length=nV) 
    end

    Omega = RPM * pi/30 * ones(nV)
    rho = 1.225 * ones(nV)

    x = [r; chord; theta; Rhub; Rtip; pitch; precone; Vinf; Omega; rho]
    y = zeros(2*nV)



    # ------- standard setup -----------

    
    # parameters that passthrough: airfoils, B, turbine, n, nV
    function ccbladewrapper!(y, x)

        r = x[1:n]
        chord = x[n+1:2*n]
        theta = x[2*n+1:3*n]
        Rhub = x[3*n+1]
        Rtip = x[3*n+2]
        pitch = x[3*n+3]
        precone = x[3*n+4]
        idx = 3*n+4
        Vinf = x[idx+1:idx+nV]
        Omega = x[idx+nV+1:idx+2*nV]
        rho = x[idx+2*nV+1:idx+3*nV]

        rotor = Rotor(Rhub, Rtip, B, turbine=turbine, precone=precone)
        sections = Section.(r, chord, theta, affunc)
        ops = simple_op.(Vinf', Omega', r, rho', pitch=pitch)

        outputs = solve.(Ref(rotor), sections, ops)

        for i = 1:nV
            T, Q = thrusttorque(rotor, sections, outputs[:, i])
            y[i] = T
            y[i+nV] = Q
        end

        return nothing
    end

    return x, y, ccbladewrapper!
end

# -------- compare forward AD and sparse forward AD ---------

nVvec = [1, 2, 4, 8, 16, 32, 64, 128, 256]
timead = zeros(9)
timeadsparse = zeros(9)
timeFDsparse = zeros(9)

for ii = 1:9
    nV = nVvec[ii]

    x, y, func = setup(nV)
    config = ForwardDiff.JacobianConfig(func, y, x)  #, ForwardDiff.Chunk{nChunk}())
    J = zeros(length(y), length(x))

    t = @benchmark ForwardDiff.jacobian!($J, $func, $y, $x, $config)
    timead[ii] = median(t).time*1e-9

    Jsparse = SparseArrays.sparse(J)  
    colors = SparseDiffTools.matrix_colors(Jsparse)
    cache = SparseDiffTools.ForwardColorJacCache(func, x, dx=y, colorvec=colors, sparsity=Jsparse)  # nChunk, 

    t = @benchmark SparseDiffTools.forwarddiff_color_jacobian!($Jsparse, $func, $x, $cache)
    timeadsparse[ii] = median(t).time*1e-9

    JsparseFD = copy(Jsparse)
    sparsecache = FiniteDiff.JacobianCache(x, Val{:forward}, colorvec=colors, sparsity=JsparseFD)

    t = @benchmark FiniteDiff.finite_difference_jacobian!($JsparseFD, $func, $x, $sparsecache)
    timeFDsparse[ii] = median(t).time*1e-9


    println(nV)
end


using PyPlot
rc("axes.spines", right=false, top=false)
rc("font", size=14.0)
rc("legend", frameon=false)

figure()
plot(nVvec, timead, "-o", color="#348ABD")
plot(nVvec, timeadsparse, "-o", color="#A60628")
xlabel("\\# inflow conditions")
ylabel("Jacobian time (s)")
legend(["AD", "AD w/ coloring"])
# savefig("adscaling.pdf")


# NOTE: the finite difference with coloring speed has improved since the paper was submitted.
figure()
plot(nVvec, timeFDsparse, "-o", color="#348ABD")
plot(nVvec, timeadsparse, "-o", color="#A60628")
xlabel("\\# inflow conditions")
ylabel("Jacobian time (s)")
legend(["FD w/ coloring", "AD w/ coloring"])
# savefig("adfd.pdf")


timead ./ timeadsparse
timeFDsparse ./ timeadsparse