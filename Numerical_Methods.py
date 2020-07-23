from IPython.display import Image
from IPython.core.display import HTML
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import sympy as sp
from sympy.plotting import plot
import operator
from scipy.special import roots_laguerre

import timeit

np.seterr(all="ignore")

# Returns a list of points (2-tuples) for each solution to the system of equations.
# If the nullclines are on top of each other, this method may produce unexpected results.
def nullcline_intersection(sys, r):
    x, y = sp.symbols('x, y')
    eq1 = sys(x, y, r)[0]
    eq2 = sys(x, y, r)[1]
    sols = sp.solve([eq1, eq2], [x, y])
    return np.array(sols,dtype=float)

def plot_nullclines(sys, linx, liny):
    t, x, y = sp.symbols('t, x, y')
    eq1 = sys[0](t, x, y)
    eq2 = sys[1](t, x, y)
    
    nullcline_x = sp.solve(eq2,x )[0] #x = f(y)
    nullclines = plot(nullcline_x)
    
    return nullclines
    

# sys is as above, x_window and y_window are lists/tuples containing the ranges for the window.
# ex: plot_flow(sys, [0, 10], [5, 15]) goes from 0 to 10 on the x axis and 5 to 15 on the y axis.
def plot_flow(sys, x_window, y_window, num_arrows, t=0):
    
    linx = np.linspace(x_window[0], x_window[1], num_arrows)
    liny = np.linspace(y_window[0], y_window[1], num_arrows)
    X, Y = np.meshgrid(linx, liny)
    
    Xa = sys[0](t, X, Y)
    Ya = sys[1](t, X, Y)
    
    n = np.sqrt(Xa*Xa + Ya*Ya)
    plt.title('Slope Field and Nullcline Intersections');
    plt.quiver(X, Y, Xa/n, Ya/n, color='g');
    
    #intersections = nullcline_intersection(sys)
    #for pt in intersections:
    #    print(f'Intersection at: {pt}')
    #    plt.plot([0], pt[1], 'bo');
        
    
    #nullcline_y = sp.solve(Xa,x)[0] #y = f(x)
    #print(nullcline_y)       
    #p1 = plot_nullclines(sys, linx, liny)
    
        
# Generates the numerical solution to an Initial Value Problem. Returns a tuple of all the times that the system was sampled at
# and a list of the corresponding states.
# sys    - list of 2 functions to solve
# init   - list of initial values for the system
# tspan  - list of beginning time and ending time
# h      - step size
# stopfn - a function that takes two arguments, time and the current state of the system. When the lambda returns a truthy value 
#          the solver will halt.
def solve_IVP(sys, init, tspan, h=None, is_forward=None):
    
    def compare(a, relate, b):
        return relate(a,b)
      
    # Check for span errors
    if tspan[0] == tspan[1]:
        print('Please make the timespan a list of two distinct values.')
    
    def wrap(sys):
        def F(t,state):
            return np.array([sys[0](t, state[0], state[1]), sys[1](t, state[0], state[1])])
        return F
    
    def rk4(t,y,h,func):
        k1 = h*func(t,y)
        k2 = h*func(t+0.5*h,y+0.5*k1)
        k3 = h*func(t+0.5*h,y+0.5*k2)
        k4 = h*func(t+h,y+k3)
        return (k1 + 2*k2 + 2*k3 + k4) / 6
    
    def rk4_solve(t0,tf,h,y0,func,relation):
        # print('(t0,tf,h,y0,func,stopfn)')
        # print(t0,tf,h,y0,func,stopfn)
        tv, yv, = [], [] # Results
        t, y = t0, y0 # Current state
        
        while (compare(t,relation,tf)):
            y += rk4(t,y,h,func) 
            t += h
            yv.append(y.copy())
            tv.append(t)
        return np.array(tv), np.array(yv)
    
    return rk4_solve(tspan[0], tspan[1], h, init, wrap(sys), is_forward)


def plot_IVP(sys, init, tspan, h=None, is_forward=None):
    # Choose the missing parameters
    forward = tspan[1] > tspan[0]
    if is_forward == None:    
        is_forward = operator.lt if forward else operator.gt
        #print(f'Choosing stopping method {"forward" if forward else "backward"}.')
    if h == None:
        h = .01 if forward else -.01
        #print(f'Choosing step size {h}.')
    t,v = solve_IVP(sys, init, tspan, h, is_forward)
    plt.plot(v[:,0],v[:,1],'y--');
    #plt.plot(init[0], init[1], 'r.')
    return t,v

# Taylor approximation at x0 of the function 'function'. The degree of the resulting polynomial is n.
def taylor(function,x0,n):
    x, y = sp.symbols('x, y')
    eq = function(x, y)
    
    def factorial(n):
        return 1 if n <= 0 else n*factorial(n-1)
    i = 0
    p = 0
    while i <= n:
        p = p + (eq.diff(x,i).subs(x,x0))/(factorial(i))*(x-x0)**i
        i += 1
    return sp.simplify(p)