# **Design Specification for Urban Traffic Signal Control Environment**

This document outlines the Markov Decision Process (MDP) formalization of the Urban Traffic Signal Control environment.

**Author**: *Vishal Pandey* and *Debasmita Biswas*
**Date**: 2025-08-05  
**Version**: 1.0  

---

## 1. State Vector Definition

Let there be $N$ intersections and $L$ lanes per intersection (total $M = N \times L$ lanes). At each control step $t$, the state vector $s_t \in \mathbb{R}^{d}$ is defined as:

- **Queue lengths**:  
  $$
  q_t = [q^1_t, q^2_t, \dots, q^M_t]
  $$
  where $q^i_t$ is the number of vehicles waiting in lane $i$ at time $t$.

- **Arrival rates**:  
  $$
  \lambda_t = [\lambda^1_t, \lambda^2_t, \dots, \lambda^M_t]
  $$
  where $\lambda^i_t$ is the estimated arrival rate (vehicles per second) to lane $i$ at time $t$.

- **Current green durations**:  
  $$
  g_t = [g^1_t, g^2_t, \dots, g^N_t]
  $$
  where $g^j_t$ is the current green-time allocation (in seconds) for intersection $j$ during this control interval.

Thus, the full state vector is:

$$
s_t = \bigl[q_t, \lambda_t, g_t\bigr] \in \mathbb{R}^{2M + N}
$$

---

## 2. Action Parameterization

At each control interval, the agent issues a continuous adjustment $a_t$ to the nominal green durations for each intersection:

$$
a_t = [\Delta g^1_t, \Delta g^2_t, \dots, \Delta g^N_t], \quad \Delta g^j_t \in [-\delta_{\max}, \delta_{\max}]
$$

The applied green times for the next interval are:

$$
g^j_{t+1} = \mathrm{clip}\bigl(g^j_t + \Delta g^j_t, g_{\min}, g_{\max}\bigr)
$$

where:

- $g_{\min}$ and $g_{\max}$ are lower/upper bounds on green durations per phase.  
- $\delta_{\max}$ is the maximum allowed adjustment magnitude.

---

## 3. Transition Equations

The dynamics evolve over a fixed control interval $\Delta t$ (e.g., 60 s). For each lane $i$:

1. **Arrival process**  
   $$
   \lambda^i_{t+1} = f_\lambda\bigl(\lambda^i_t\bigr) + \epsilon^i_t, \quad \epsilon^i_t \sim \mathcal{N}(0,\sigma^2)
   $$

2. **Service rate**  
   For intersection $j$, given green time $g^j_t$, the service (throughput) rate per lane in that intersection is:

   $$
   \mu^i_t = \frac{g^j_t}{C_t} \times s_{\mathrm{sat}}
   $$

   where:

   - $C_t = \sum_{k=1}^N g^k_t + N \times t_{\mathrm{all\_red}}$ is the cycle length (sum of all greens plus all-red/amber durations).  
   - $s_{\mathrm{sat}}$ is the saturation flow rate (vehicles per second of green).

3. **Queue update**  
   $$
   q^i_{t+1} = \max\left\{0,\;q^i_t + \lambda^i_t \Delta t - \mu^i_t \Delta t\right\}
   $$

This deterministic update is supplemented by stochastic arrivals from step 1.

---

## 4. Reward Function & Discount Factor

We design the reward to capture network performance, balancing delay minimization and fairness:

- **Total waiting time penalty**:  
  $$
  r^{(1)}_t = -\sum_{i=1}^{M} q^i_{t+1}
  $$

- **Queue imbalance penalty**:  
  $$
  r^{(2)}_t = -\kappa \sum_{i=1}^{M} \left|q^i_{t+1} - \bar{q}_{t+1}\right|, \quad \bar{q}_{t+1} = \frac{1}{M} \sum_{i} q^i_{t+1}
  $$

- **Throughput bonus (optional)**:  
  $$
  r^{(3)}_t = +\alpha \sum_{i=1}^{M} \min\left(q^i_t, \mu^i_t \Delta t\right)
  $$

We combine these terms into a scalar reward:

$$
r_t = r^{(1)}_t + r^{(2)}_t + r^{(3)}_t
$$

Finally, we apply a discount factor $\gamma \in [0,1]$ to future rewards in the RL algorithm, typically $\gamma \approx 0.99$.

---