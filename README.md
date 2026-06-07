# Physics-Informed Neural Networks (PINNs)

Link to blog post, which goes into more about the mass-spring-damper system that I am using the PINNs to analyse: [link](https://bentay85.github.io/2026/06/07/pinn.html)

Steve Brunton's video about Physics-Informed Neural Networks: [link](https://www.youtube.com/watch?v=-zrY7P2dVC4)  

Ben Moseley's Physics-Informed Neural Networks Lecture: [link](https://www.youtube.com/watch?v=B-t-w8wAIiY)

### Environment Setup

This repository uses uv to manage dependencies so you can download it from the [link](https://github.com/astral-sh/uv) add add it to PATH.

Run the following commands:

```
git clone: https://github.com/bentay85/pinn.git
cd pinn
uv sync
uv run 1_Forward_Euler.py
```

### Scripts

All the scripts are in the root directory so you can run them one by one. The graphs that are generated are saved to a graphs directory. Technical details about what each of the scrips do are discussed within the my [blog post](https://bentay85.github.io/2026/06/07/pinn.html).  

* 1_Forward_Euler.py  
* 2_pinn_simulate.py  
* 3_pinn_data.py
* 4_pinn_unknown_parameters.py
