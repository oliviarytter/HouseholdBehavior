import numpy as np
from scipy.optimize import minimize,  NonlinearConstraint
import warnings
warnings.filterwarnings("ignore", message="delta_grad == 0.0. Check if the approximated function is linear.") # turn of annoying warning

from EconModel import EconModelClass, jit

from consav.grids import nonlinspace
from consav.linear_interp import interp_2d

class DynLaborModelClass(EconModelClass):

    def settings(self):
        """ fundamental settings """

        pass

    def setup(self):
        """ set baseline parameters """

        # unpack
        par = self.par

        par.T = 10 # time periods
        
        # preferences
        par.rho = 0.99 # discount factor

        par.beta = 1.0 # weight on labor dis-utility
        par.eta = -2.5 # CRRA coefficient
        par.gamma = 2.5 # curvature on labor hours 

        # income
        par.alpha = 0.1 # human capital accumulation 
        par.w = 1.0 # wage base level
        par.tau = 0.12 # labor income tax
        par.tau_a = 0.0 # tax on wealth 

        # human capital depreciation shock
        par.delta = 0.2   # size of depreciation (20%)
        par.pk    = 0.6   # probability that depreciation occurs

        # disability
        par.kappa = 0.5   # wage reduction when disabled
        par.b     = 0.2   # disability benefit
        par.pd    = 0.1   # disability prob (doubles if already disabled)
        par.Nd    = 2     # disability states: {0, 1}

        # saving
        par.r = 0.03 # interest rate

        # grids
        par.a_max = 5.0 # maximum point in wealth grid
        par.a_min = -5.0 # minimum point in wealth grid
        par.Na = 40 # number of grid points in wealth grid 
        
        par.k_max = 15.0 # maximum point in wealth grid
        par.Nk = 10 # number of grid points in wealth grid    

        # simulation
        par.simT = par.T # number of periods
        par.simN = 1_000 # number of individuals



    def allocate(self):
        """ allocate model """

        # unpack
        par = self.par
        sol = self.sol
        sim = self.sim

        par.simT = par.T
        
        # a. asset grid
        par.a_grid = nonlinspace(par.a_min,par.a_max,par.Na,1.1)

        # b. human capital grid
        par.k_grid = nonlinspace(0.0,par.k_max,par.Nk,1.1)

        par.d_grid = np.arange(par.Nd)          # [0, 1]

        shape = (par.T,par.Na,par.Nk,par.Nd)    # solution arrays now carry disability
        sol.c = np.nan + np.zeros(shape)
        sol.h = np.nan + np.zeros(shape)
        sol.V = np.nan + np.zeros(shape)

        # d. simulation arrays
        shape = (par.simN,par.simT)
        sim.c = np.nan + np.zeros(shape)
        sim.h = np.nan + np.zeros(shape)
        sim.a = np.nan + np.zeros(shape)
        sim.k = np.nan + np.zeros(shape)

        sim.budget = np.nan + np.zeros(shape)
        sim.util = np.nan + np.zeros(shape)

        # e. initialization
        np.random.seed(3500)
        sim.a_init = np.fmax(0.0 , np.random.normal(size=par.simN))
        sim.k_init = np.fmax(0.0 , 0.5*np.random.normal(size=par.simN))

        # depreciation shocks: True where human capital depreciates this period
        sim.delta_shock = np.random.uniform(size=(par.simN,par.simT)) < par.pk
        sim.d = np.nan + np.zeros((par.simN,par.simT))                 # disability status
        sim.disability_shock = np.random.uniform(size=(par.simN,par.simT))



    ############
    # Solution #
    def solve(self):
        par = self.par
        sol = self.sol

        for t in reversed(range(par.T)):                                # unchanged
            for i_a,assets in enumerate(par.a_grid):                    # unchanged
                for i_k,capital in enumerate(par.k_grid):               # unchanged
                    for i_d,disabled in enumerate(par.d_grid):          # NEW loop over disability
                        idx            = (t,i_a,i_k,i_d)
                        idx_last       = (t+1,i_a,i_k,i_d)
                        idx_prev_asset = (t,i_a-1,i_k,i_d)

                        if t==par.T-1:
                            obj = lambda x: self.obj_last(x[0],assets,capital,disabled)
                            hours_min = np.fmax(-assets/self.wage_func(capital,disabled)+1.0e-5, 0.0)
                            init_h = np.maximum(hours_min,2.0) if i_a==0 else np.array([sol.h[idx_prev_asset]])
                            res = minimize(obj,init_h,bounds=((hours_min,np.inf),),method='L-BFGS-B')
                            sol.c[idx] = self.cons_last(res.x[0],assets,capital,disabled)
                            sol.h[idx] = res.x[0]
                            sol.V[idx] = self.util(sol.c[idx],sol.h[idx])
                        else:
                            obj = lambda x: - self.value_of_choice(x[0],x[1],assets,capital,disabled,t)
                            init = np.array([sol.c[idx_last],sol.h[idx_last]])
                            res = minimize(obj,init,bounds=((1e-6,np.inf),(0.0,np.inf)),method='L-BFGS-B',tol=1.0e-10)
                            sol.c[idx] = res.x[0]
                            sol.h[idx] = res.x[1]
                            sol.V[idx] = -res.fun

    # last period
    def cons_last(self,hours,assets,capital,disabled):
        par = self.par
        income = self.wage_func(capital,disabled) * hours
        wealth_tax_rate = par.tau_a * (assets > 0.0)
        cons = (1.0-wealth_tax_rate)*assets + income + par.b*disabled   # +benefit in last period too
        return cons

    def obj_last(self,hours,assets,capital,disabled):
        cons = self.cons_last(hours,assets,capital,disabled)
        return - self.util(cons,hours)

    # earlier periods
    def value_of_choice(self,cons,hours,assets,capital,disabled,t):

        # a. unpack
        par = self.par
        sol = self.sol

        # b. penalty for violating bounds. 
        penalty = 0.0
        if cons < 0.0:
            penalty += cons*1_000.0
            cons = 1.0e-5
        if hours < 0.0:
            penalty += hours*1_000.0
            hours = 0.0

        # c. utility from consumption
        util = self.util(cons,hours)
        
        # d. EXPECTED continuation value over the human-capital depreciation shock
        a_next = self.wealth_trans(assets,capital,disabled,hours,cons)
        k_next_no   = capital + hours
        k_next_depr = (1.0-par.delta)*capital + hours

        prob_dis = par.pd * (1.0 + disabled)   # disabled next period: pd if healthy, 2*pd if disabled

        # double expectation: over disability (outer) then depreciation (inner)
        EV_next = 0.0
        for i_d_next, prob_d in ((0, 1.0-prob_dis), (1, prob_dis)):
            V_next = sol.V[t+1,:,:,i_d_next]                       # value slice at this disability state
            V_no   = interp_2d(par.a_grid,par.k_grid,V_next,a_next,k_next_no)
            V_depr = interp_2d(par.a_grid,par.k_grid,V_next,a_next,k_next_depr)
            EV_next += prob_d * (par.pk*V_depr + (1.0-par.pk)*V_no)

        # e. return value of choice (including penalty)
        return util + par.rho*EV_next + penalty


    def util(self,c,hours):
        par = self.par

        return (c)**(1.0+par.eta) / (1.0+par.eta) - par.beta*(hours)**(1.0+par.gamma) / (1.0+par.gamma) 

    def wage_func_before_tax(self,capital,disabled):
        par = self.par
        return par.w * (1.0 + par.alpha * capital) * (1.0 - par.kappa*disabled)   # wage cut if disabled

    def wage_func(self,capital,disabled):
        return (1.0 - self.par.tau) * self.wage_func_before_tax(capital,disabled)

    def wealth_trans(self,assets,capital,disabled,hours,cons):
        par = self.par
        income = self.wage_func(capital,disabled) * hours
        wealth_tax_rate = par.tau_a * (assets > 0.0)
        a_next = (1.0+par.r)*((1.0-wealth_tax_rate)*assets + income + par.b*disabled - cons)  # +benefit
        return a_next

    ##############
    # Simulation #
    def simulate(self):

        # a. unpack
        par = self.par
        sol = self.sol
        sim = self.sim

        # b. loop over individuals and time
        for i in range(par.simN):

            # i. initialize states
            sim.a[i,0] = sim.a_init[i]
            sim.k[i,0] = sim.k_init[i]
            sim.d[i,0] = 0.0  

            for t in range(par.simT):
                d_now = int(sim.d[i,t])                         # current disability status (0 or 1)

                # policy at the current disability slice, interpolated over (a,k)
                sim.c[i,t] = interp_2d(par.a_grid,par.k_grid,sol.c[t,:,:,d_now],sim.a[i,t],sim.k[i,t])
                sim.h[i,t] = interp_2d(par.a_grid,par.k_grid,sol.h[t,:,:,d_now],sim.a[i,t],sim.k[i,t])
                sim.util[i,t] = self.util(sim.c[i,t],sim.h[i,t])

                # revenue: labour tax (disability-adjusted wage) + wealth tax - benefit paid out
                sim.budget[i,t] = (par.tau*self.wage_func_before_tax(sim.k[i,t],d_now)*sim.h[i,t]
                                + par.tau_a*sim.a[i,t]*(sim.a[i,t]>0.0)
                                - par.b*d_now)

                if t<par.simT-1:
                    sim.a[i,t+1] = self.wealth_trans(sim.a[i,t],sim.k[i,t],d_now,sim.h[i,t],sim.c[i,t])

                    # human capital depreciation
                    if sim.delta_shock[i,t]:
                        sim.k[i,t+1] = (1.0-par.delta)*sim.k[i,t] + sim.h[i,t]
                    else:
                        sim.k[i,t+1] = sim.k[i,t] + sim.h[i,t]

                    # disability Markov draw: disabled next period if shock below the threshold
                    prob_dis = par.pd * (1.0 + d_now)           # pd if healthy, 2*pd if disabled
                    sim.d[i,t+1] = 1.0 if sim.disability_shock[i,t] < prob_dis else 0.0


