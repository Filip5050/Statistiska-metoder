import numpy as np
from scipy.stats import t

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
        # Beräknar t-värden och p-värden för varje koefficient
        variance_beta = np.diagonal(self.variance() * np.linalg.inv(self.X.T @ self.X))
        std = np.sqrt(variance_beta)
        t_values = self.beta / std
        p_values = [2 * (1 - t.cdf(np.abs(t_val), df=self.n - self.d)) for t_val in t_values]
        return t_values, p_values
    
    def relevance(self):
        # Beräknar R^2
        SST = self.SST()
        SSE = self.SSE()
        return 1 - (SSE / SST)
