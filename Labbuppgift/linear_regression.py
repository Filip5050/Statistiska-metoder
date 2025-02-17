import numpy as np
from scipy.stats import f

class LinearRegression: 
    def __init__(self, X, y):
        # Skapar en X (oberoende variabel) och lägger till en kolumn med ettor 
        # för att inkludera interceptet.
        self.X = np.hstack((np.ones((X.shape[0], 1)), X))
        self.y = y  # Skapar y (beroende variabel)
        self.beta = None  # Skapar Beta (koefficient) variabel och sätter den till None så länge

    @property
    def d(self):  # d är antal parametrar med intercept
        return self.X.shape[1]
    
    @property
    def n(self): # n är antal observationer
        return self.X.shape[0]
    
    def Koefficient(self):
        # Beräknar koefficienterna och intercepten
        self.beta = np.linalg.inv(self.X.T @ self.X) @ self.X.T @ self.y

    def SST(self):
        #Beräknda Total Sum of Squares
        return np.sum((self.y - np.mean(self.y))**2)

    def SSR(self):
        #Beräknar Regression Sum of Squares
        ŷ = self.X @ self.beta
        return np.sum((ŷ - np.mean(self.y))**2)
    
    def SSE(self):
        # eräknar Error Sum of Squares
        ŷ = self.X @ self.beta
        return np.sum((self.y - ŷ)**2)

    def variance(self):
        # Beräknar variansen
        SSE = self.SSE()
        return SSE / (self.n - self.d)
    
    def standard_deviation(self):
        # Standard devition får vi genom att ta roten ur på variansen
        return np.sqrt(self.variance())
    
    def significance(self):
        
        d = self.d - 1
        df_res = self.n - self.d
        SSR = self.SSR()
        SSE = self.SSE()
        sigma2 = SSE / df_res
        F_stat = (SSR / d) / sigma2
        p_value = f.sf(F_stat, d, df_res)
        return F_stat, p_value

    
    def relevance(self):
        # Beräknar R^2
        SST = self.SST()
        SSE = self.SSE()
        return 1 - (SSE / SST)
