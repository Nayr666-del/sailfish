import numpy as np
from scipy.optimize import root_scalar

class ML_Boundaries():
    def __init__(self,
        r_buffer,
        em,
        t_list,
        ):
        self.r_buffer = r_buffer
        self.em = em
        self.t_list = t_list


    def epi(self,r0, t, em):
        return r0 + r0 * em * (1 - np.cos((1 - 2 * em) * t / r0**1.5))
    
    def S(self,r0,t,em):
        return np.abs(1 + 2 * em * (1 - np.cos((1-2*em) * t / r0**1.5)) - 3/2 * em * r0**(-1.5) * t * np.sin((1-2*em) * t / r0**1.5))
    
    @property
    def shocked_conditions(self):
        r0_vals = np.linspace(self.r_buffer * 1.1, self.r_buffer / (1+2*self.em) * 0.9, 1000)  # Avoid r0=0 to prevent division by zero
        tol = 1e-5
        SS_new = np.zeros_like(self.t_list)
        for count, t in enumerate(self.t_list):
            found_roots = []
            # Search for intervals where g(r0) changes sign
            def g(r0):
                return self.epi(r0, t, self.em) - self.r_buffer
            for i in range(len(r0_vals) - 1):
                r0a, r0b = r0_vals[i], r0_vals[i+1]
                if g(r0a) * g(r0b) < 0:
                    try:
                        sol = root_scalar(g, bracket=[r0a, r0b], method='brentq', xtol=tol)
                        root = sol.root
                        # Avoid duplicates
                        if not any(np.isclose(root, x, atol=tol) for x in found_roots):
                            found_roots.append(root)
                    except ValueError:
                        pass  # Skip if root finding failed
            for r_roots in found_roots:
                SS_new[count] += 1.89e-4 * r_roots ** (-3/5) / self.S(r_roots,t,self.em) #Currently hard-coded


        return SS_new
            
            