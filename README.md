# Physics-Informed Neural Networks (PINNs)

This repository explores the implementation and application of **Physics-Informed Neural Networks (PINNs)**. 

Traditional deep learning models rely exclusively on data-driven pattern recognition, often failing to generalize outside their training distribution or violating fundamental conservation laws. PINNs mitigate these limitations by embedding known governing physical laws directly into the neural network's optimization objective.

### Core Mechanics

By leveraging neural networks as universal function approximators and utilizing the automatic differentiation capabilities native to PyTorch, PINNs enforce differential equations and initial/boundary conditions directly within the loss function.

* **Functional Approximation:** The neural network acts as a continuous function approximator to represent the solution of a physical system.
* **Automatic Differentiation:** Partial derivatives are computed via PyTorch's autograd engine, giving us a convenient way of introducing the underlying physics into the training. 
* **Composite Loss Formulation:** The objective function is regularized by penalizing deviations from both empirical data and underlying physical laws: L = L(data) + L(physics) + L(initial/boundary)

This hybrid approach restricts the network's search space to physically consistent solutions, significantly improving data efficiency, generalization, and robustness in data-scarce regimes.  

### Problem System

I chose the a mass-spring-damper system without external forcing as it is a simple system governed by a second-order linear ordinary differential equation (ODE) derived from Newton's Second Law. For derivation of the solution you can refer to link [here](https://math.libretexts.org/Courses/Cosumnes_River_College/Math_420%3A_Differential_Equations_(Breitenbach)/06%3A_Applications_of_Linear_Second_Order_Equations/6.02%3A_Spring-Mass_Problems_(With_Damping))

```
m * d²x/dt² + μ * dx/dt + k * x = 0
```  

Given Parameters:
* Mass (m) = 1 kg
* Damping coefficient (μ) = 4 N·s/m
* Spring stiffness (k) = 400 N/m

Initial Conditions:
* Initial displacement: x(0) = 1
* Initial velocity: dx/dt(0) = 0  

An underdammped system was chosen which would exhibit oscillatory behaviour to make it more challenging to model. Substituting the parameters into the governing equation yields:

```
d²x/dt² + 4 * dx/dt + 400 * x = 0
```
The analytic solution to this problem has the form:

```
x(t) = e^(-α*t) * [A * cos(ω_d * t) + B * sin(ω_d * t)]  

OR    

x(t) = C * e^(-α*t) * cos(ω_d * t - φ)
```  
Using the quadratic formula, r² + 4*r + 400 = 0, to solve for r: 

r = -2 ± 19.900 * i  

α = 2
ω_d = 19.900

Substituting in the initial conditions,

A = 1  
B = (α * A) / ω_d = (2 * 1) / 19.900 = 0.10050
C = √(A² + B²) = 1.0050  
φ = atan(B / A) = 0.10017

The anaytical solution is:  
```
x(t) = 1.0050 * e^(-2 * t) * cos(19.900 * t - 0.10017)
```  

