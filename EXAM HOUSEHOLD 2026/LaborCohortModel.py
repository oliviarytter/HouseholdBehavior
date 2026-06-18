import numpy as np
from scipy.optimize import minimize

from EconModel import EconModelClass

from consav.grids import nonlinspace
from consav.linear_interp import interp_1d
from consav.quadrature import log_normal_gauss_hermite

class LaborCohortModelClass(EconModelClass):

    def settings(self):
        """ fundamental settings """

        pass

    def setup(self):
        """ set baseline parameters """

        # unpack
        par = self.par

        par.T = 20 # time periods

        # grids       
        par.k_max = 20.0 # maximum point in human capital grid
        par.Nk = 50 # number of grid points in human capital grid    

        par.Nb = 5 # number of birth cohorts

        par.Nshock = 5 # number of shocks for integration of human capital shocks
        par.sigma = 0.1 # standard deviation of human capital shocks

        par.num_years = par.T + par.Nb - 1 # number of years in simulation, needed for aggregate shock process
        
        # preferences
        par.beta = 0.97 # discount factor

        par.rho = 1.5 # CRRA coefficient
        par.phi = 3.0 # disutility of labor parameter
        par.eta = 2.0 # inverse Frisch elasticity of labor supply

        # wages
        par.alpha = np.linspace(0.7,1.2,par.Nb) # constant in wage function
        par.gamma = 0.1 # return to human capital in wage equation

        par.delta = 0.1 # human capital depreciation
        par.sigma = 0.1 # standard deviation of human capital shocks

        par.lambda_0 = -0.001 # constant in aggregate shock process

        # policy parameters
        par.tau = 0.3 # tax rate on labor income

        # Added in 4.b
        par.pi_0 = 0.5 # constant in unemployment benefits
        par.pi_1 = 0.0 # additional old-age subsidy (0 = off, recovers the Q3 model)
        par.retire_age  = 9 # age threshold 
        par.retire_year = 7 # year threshold 

        # simulation
        par.simT = par.T # number of periods
        par.simN = 50_000 # number of individuals


    def allocate(self):
        """ allocate model """

        # unpack
        par = self.par
        sol = self.sol
        sim = self.sim

        par.simT = par.T
        
        # a. human capital grid
        par.k_grid = nonlinspace(0.0,par.k_max,par.Nk,1.1)

        # b. birth cohort grid
        par.b_grid = np.arange(par.Nb)

        # c. human capital shocks
        par.shock_grid,par.shock_prob = log_normal_gauss_hermite(par.sigma,par.Nshock)

        # c. solution arrays
        shape = (par.T,par.Nb,par.Nk)
        sol.h = np.nan + np.zeros(shape)
        sol.V = np.nan + np.zeros(shape)

        # d. simulation arrays
        shape = (par.simN,par.simT)
        sim.h = np.nan + np.zeros(shape)
        sim.k = np.nan + np.zeros(shape)
        sim.b = np.zeros(shape,dtype=int)
        sim.y = np.zeros(shape,dtype=int)
        sim.t = np.zeros(shape,dtype=int)
        # Consumption 
        sim.c = np.nan + np.zeros(shape)  
        sim.G = np.zeros(shape)  


        # e. initialization and random draws for simulation
        np.random.seed(2026)
        ktilde = np.random.normal(0.0,1.0,par.simN) # initial human capital shock
        sim.k_init = np.fmin(np.fmax(ktilde,0.0) , 3.0)
        sim.b_init = np.int_(np.random.choice(par.b_grid,par.simN)) # initial birth cohort, uniformly distributed

        sim.draw_shock = np.random.choice(par.shock_grid,size=shape,p=par.shock_prob) # draw shocks for simulation of human capital transitions


    ############
    # Solution #
    def solve(self,do_accurate=False):

        # a. unpack
        par = self.par
        sol = self.sol
        
        # b. loop backwards (over all periods)
        for t in reversed(range(par.T)):

            # i. loop over state variables: birth cohort and human capital
            for cohort in par.b_grid: # cohort is integer starting in zero. Thus it is also used as index
                for i_k,capital in enumerate(par.k_grid):
                    idx = (t,cohort,i_k)

                    # ii. find optimal consumption and hours at this level of wealth in this period t.

                    if t==par.T-1: # last period
                        obj = lambda x: - self.util(x[0],t,cohort,capital)

                        init = np.array([0.5]) # initial guess for hours

                    else:
                        
                        # objective function: negative since we minimize
                        obj = lambda x: - self.value_of_choice(x[0],t,cohort,capital)  

                        idx_last = (t+1,cohort,i_k)
                        init = np.array([sol.h[idx_last]]) # use next period's hours as initial guess for current period's hours

                    # call optimizer
                    res = minimize(obj,init,bounds=((1.0e-6,1.0),),method='L-BFGS-B',tol=1.0e-10)
                    opt_h = res.x[0]
                    opt_val = res.fun

                    # check corner solutions - ALSO changed in 4.b
                    if self.nonemplyment_income(0.0,t,self.year_func(t,cohort)) > 0.0: # can only be optimal to not work if there are nonemployment benefits available
                        obj0 = obj(np.array([0.0]))
                        if obj0 < opt_val:
                            opt_h = 0.0
                            opt_val = obj0

                    obj1 = obj(np.array([1.0]))
                    if obj1 < opt_val:
                        opt_h = 1.0
                        opt_val = obj1

                    # store results
                    sol.h[idx] = opt_h
                    sol.V[idx] = -opt_val

    
    # Utility and value of choice functions
    def util(self,hours,t,cohort,capital):
        par = self.par

        cons = self.cons_budget(hours,t,cohort,capital)
        util_cons = (cons**(1.0-par.rho))/(1.0-par.rho)
        util_labor = par.phi * hours**(1.0+par.eta) / (1.0+par.eta)

        return util_cons - util_labor

    def value_of_choice(self,hours,t,cohort,capital):

        # a. unpack
        par = self.par
        sol = self.sol

        # b. penalty for violating bounds. 
        penalty = 0.0
        if hours < 0.0:
            penalty += hours*1_000.0
            hours = 0.0
        elif hours > 1.0:
            penalty += (1.0-hours)*1_000.0
            hours = 1.0

        # c. utility from consumption
        util = self.util(hours,t,cohort,capital)
        
        # d. expected continuation value from savings
        V_next = sol.V[t+1,cohort]
        EV_next = 0.0
        for i_shock,shock in enumerate(par.shock_grid):

            k_next = self.human_capital_trans(capital,hours,shock)
            V_next_interp = interp_1d(par.k_grid,V_next,k_next)

            EV_next += par.shock_prob[i_shock] * V_next_interp

        # e. return value of choice (including penalty)
        return util + par.beta*EV_next + penalty

    
    # states and budget equations
    def cons_budget(self,hours,t,cohort,capital):
        par = self.par

        year = self.year_func(t,cohort)
        labor_income = self.wage_func(cohort,capital,year) * hours
        
        unemployment_benefits = self.nonemplyment_income(hours,t,year) # pass age and year for the old-age subsidy

        cons = (1-par.tau)*labor_income + unemployment_benefits
        return cons

    def wage_func(self,cohort,capital,year):
        par = self.par

        individual_wage = np.exp(par.alpha[cohort] + par.gamma * capital)
        aggregate_shock = self.aggregate_shock_func(year)

        return individual_wage * aggregate_shock
    
    def human_capital_trans(self,capital,hours,shock):
        par = self.par

        return ((1.0-par.delta)*capital + hours) * shock
    
    def aggregate_shock_func(self,year):
        par = self.par

        return 1.0 + par.lambda_0 * year
    
    def year_func(self,t,cohort):
        return t + cohort
    
    # Adjusted in 4.b: non-employment income that depends on age and year
    def nonemplyment_income(self,hours,t,year):
        par = self.par
        # benefit is pi_0 plus the old-age top-up pi_1, paid only if old enough and late enough
        pi = par.pi_0 + par.pi_1 * (t >= par.retire_age) * (year >= par.retire_year)
        return pi * (hours < 1.0e-6)


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
            sim.k[i,0] = sim.k_init[i]
            sim.b[i,0] = sim.b_init[i]

            for t in range(par.simT):

                sim.t[i,t] = t # age
                sim.y[i,t] = self.year_func(t,sim.b[i,t]) # year

                # ii. interpolate optimal consumption and hours
                idx_sol = (t,sim.b[i,t])
                sim.h[i,t] = interp_1d(par.k_grid,sol.h[idx_sol],sim.k[i,t])
                sim.h[i,t] = np.clip(sim.h[i,t],0.0,1.0) # make sure hours are between 0 and 1
                sim.c[i,t] = self.cons_budget(sim.h[i,t],t,sim.b[i,t],sim.k[i,t])  
                wage = self.wage_func(sim.b[i,t],sim.k[i,t],sim.y[i,t])          
                sim.G[i,t] = par.tau*wage*sim.h[i,t] - self.nonemplyment_income(sim.h[i,t],t,sim.y[i,t])


                # iii. store next-period states
                if t<par.simT-1:
                    sim.k[i,t+1] = self.human_capital_trans(sim.k[i,t],sim.h[i,t],sim.draw_shock[i,t+1])
                    sim.b[i,t+1] = sim.b[i,t]  # birth cohort constant over time


