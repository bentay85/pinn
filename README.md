# **Physics-Informed Neural Networks (PINNs)**

This document outlines the implementation and application of Physics-Informed Neural Networks (PINNs).

Traditional deep learning models rely strictly on empirical data-driven pattern recognition. Consequently, they often fail to generalize outside their training distribution and can violate fundamental conservation laws. PINNs mitigate these vulnerabilities by embedding known governing physical laws directly into the neural network's Loss Function.

## **Core Mechanics**

PINNs leverage neural networks as universal function approximators while enforcing physical consistency. By utilizing the automatic differentiation capabilities native to modern deep learning frameworks (such as PyTorch's autograd), the network evaluates differential equations and initial/boundary conditions without requiring traditional numerical discretization (e.g., finite differences or finite elements).

### **1\. Functional Approximation**

The neural network acts as a continuous model, mapping independent variables (such as time, t, and space, x) to the physical state variables of the system.

### **2\. Automatic Differentiation**

Partial derivatives are computed using the chain rule via the framework's computational graph. This enables the calculation of physical residuals at arbitrary points in the domain, bypassing mesh-generation constraints.

### **3\. Composite Loss Formulation**

To restrict the optimization search space to physically consistent states, the network's objective function is formulated as a composite loss:

Total Loss \= Loss(data) \+ Loss(physics) \+ Loss(boundary)

* **Loss(data):** Penalizes deviation from observed/empirical data points.  
* **Loss(physics):** Penalizes deviations from the governing differential equations evaluated at chosen collocation points.  
* **Loss(boundary):** Enforces boundary and initial conditions.

This regularization dramatically improves data efficiency and ensures robustness in data-scarce regimes.

## **Model Problem: Mass-Spring-Damper System**

To evaluate PINN performance, we model a classical mass-spring-damper system without external excitation. This system is governed by a second-order linear ordinary differential equation (ODE) derived from Newton's Second Law:

m \* (d²x/dt²) \+ μ \* (dx/dt) \+ k \* x \= 0

### **Given Parameters**

* Mass (m) \= 1.0 kg  
* Damping coefficient (μ) \= 4.0 N·s/m  
* Spring stiffness (k) \= 400.0 N/m

### **Initial Conditions**

* Initial displacement: x(0) \= 1.0 m  
* Initial velocity: dx/dt(0) \= 0.0 m/s

Substituting these physical parameters into the governing equation yields:

d²x/dt² \+ 4 \* (dx/dt) \+ 400 \* x \= 0

This configuration describes an underdamped oscillator, which exhibits decaying harmonic motion—providing a challenging, dynamic target for neural network training.

## **Analytical Reference Solution**

To establish a baseline for verification, we derive the exact analytical solution. Details can be found in this [link](https://math.libretexts.org/Courses/Cosumnes_River_College/Math_420%3A_Differential_Equations_(Breitenbach)/06%3A_Applications_of_Linear_Second_Order_Equations/6.02%3A_Spring-Mass_Problems_(With_Damping)). The characteristic equation is:  

r² \+ 4\*r \+ 400 \= 0

Solving for the roots (r):

r \= \-2 ± i \* sqrt(396)

r ≈ \-2 ± 19.900 \* i

This yields the attenuation factor (alpha) and damped natural frequency (omega\_d):

alpha \= 2.0

omega\_d ≈ 19.900 rad/s

The general solution for an underdamped system is:

x(t) \= e^(-alpha \* t) \* \[A \* cos(omega\_d \* t) \+ B \* sin(omega\_d \* t)\]

Applying the initial conditions:

1. **At t \= 0, x(0) \= 1.0:**  
   x(0) \= e^(0) \* \[A \* cos(0) \+ B \* sin(0)\] \= A  
   A \= 1.0  
2. **At t \= 0, dx/dt(0) \= 0.0:**  
   Differentiating x(t):  
   dx/dt \= \-alpha \* e^(-alpha \* t) \* \[A \* cos(omega\_d \* t) \+ B \* sin(omega\_d \* t)\] \+ e^(-alpha \* t) \* \[-A \* omega\_d \* sin(omega\_d \* t) \+ B \* omega\_d \* cos(omega\_d \* t)\]  
   Evaluating at t \= 0:  
   0 \= \-alpha \* A \+ B \* omega\_d  
   B \= (alpha \* A) / omega\_d  
   B \= (2.0 \* 1.0) / 19.900 ≈ 0.10050

Expressing this in amplitude-phase form:

x(t) \= C \* e^(-alpha \* t) \* cos(omega\_d \* t \- phi)

Where:

* Amplitude (C) \= sqrt(A² \+ B²) \= sqrt(1.0² \+ 0.10050²) ≈ 1.0050  
* Phase angle (phi) \= arctan(B / A) \= arctan(0.10050 / 1.0) ≈ 0.10017 rad

This gives the final, exact analytical solution used for evaluating PINN accuracy:

x(t) \= 1.0050 \* e^(-2 \* t) \* cos(19.900 \* t \- 0.10017)