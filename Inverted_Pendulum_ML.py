import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.integrate import solve_ivp

# ─────────────────────────────
# IMPORT YOUR PENDULUM
# ─────────────────────────────
class InvertedPendulum:
    def __init__(self):
        # Physical Parameters
        self.g = 9.81
        self.L = 1
        self.m = 1
        self.D = 0.1

        # PID Gains
        self.Kp = 100
        self.Ki = 15
        self.Kd = 25

    def dynamics(self, t, y):
        theta, omega, integral = y

        # Target Value is = degrees
        error = 0 - theta

        # Defining the Control Input
        u_control = (self.Kp * error) + (self.Ki * integral) - (self.Kd * omega)

        # Equation of Motion
        # Ml^2* domega = mg sin alpha L - D omega -F
        d_theta = omega
        inertia = self.m * self.L**2
        d_omega = (self.g/self.L)*np.sin(theta) - (self.D/inertia)*omega + (u_control/inertia)

        d_integral = error

        return [d_theta, d_omega, d_integral]

    def simulate(self, duration=10, theta0=0.2, omega0=0.0):
        t_span = (0, duration)
        t = np.linspace(0, duration, 500)
        y0 = [theta0, omega0, 0.0]

        sol = solve_ivp(self.dynamics, t_span, y0, t_eval=t)
        return sol

# ─────────────────────────────
# STEP 1 — GENERATE TRAINING DATA
# ─────────────────────────────
def generate_data():
    pendulum = InvertedPendulum()
    all_inputs = []
    all_outputs = []

    # 20 different starting angles
    starting_angles = np.linspace(-0.5, 0.5, 20)

    for theta0 in starting_angles:
        sol = pendulum.simulate(theta0=theta0)
        theta = sol.y[0]
        omega = sol.y[1]

        for i in range(len(theta) - 1):
            inp = [theta[i], omega[i]]
            out = [theta[i+1], omega[i+1]]
            all_inputs.append(inp)
            all_outputs.append(out)

    X = torch.tensor(all_inputs, dtype=torch.float32)
    y = torch.tensor(all_outputs, dtype=torch.float32)
    print(f"Generated {len(X)} training samples")
    return X, y

# ─────────────────────────────
# STEP 2 — NEURAL NETWORK
# ─────────────────────────────
class PendulumNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(2, 64),     
            nn.ReLU(),
            nn.Linear(64, 64),    
            nn.ReLU(),
            nn.Linear(64, 2)      
        )

    def forward(self, x):
        return self.network(x)

# ─────────────────────────────
# STEP 3 — TRAINING THE NETWORK
# ─────────────────────────────
def train(X, y, epochs=1000):
    model = PendulumNet()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()
    losses = []

    for epoch in range(epochs):
        # Predict and measure error
        loss = loss_fn(model(X), y)

        # Update weights
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        if epoch % 100 == 0:
            print(f"Epoch {epoch}/{epochs} — Loss: {loss.item():.6f}")

    return model, losses
# ─────────────────────────────
# STEP 4 — SIMULATE WITH NN
# ─────────────────────────────
def simulate_with_nn(model, theta0, omega0, steps=499):
    state = torch.tensor([[theta0, omega0]], dtype=torch.float32)
    trajectory = [state.numpy()[0]]

    model.eval()
    with torch.no_grad():
        for _ in range(steps):
            state = model(state)
            trajectory.append(state.numpy()[0])

    return np.array(trajectory)
# ─────────────────────────────
# STEP 5 — COMPARE AND PLOT
# ─────────────────────────────
def compare(model, theta0=0.3):
   # Real simulation
    sol = InvertedPendulum().simulate(theta0=theta0)
    t        = sol.t
    theta_real = sol.y[0]
    omega_real = sol.y[1]

    # NN simulation
    nn_traj  = simulate_with_nn(model, theta0, 0.0, steps=len(t)-1)
    theta_nn = nn_traj[:, 0]
    omega_nn = nn_traj[:, 1]

    # Plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    ax1.plot(t, np.degrees(theta_real), label='Real (scipy)', color='steelblue', lw=2)
    ax1.plot(t, np.degrees(theta_nn),  label='Neural Network', color='tomato', lw=2, linestyle='--')
    ax1.axhline(0, color='gray', linestyle=':', alpha=0.5)
    ax1.set_ylabel('Angle (degrees)')
    ax1.set_title('Inverted Pendulum — PID Simulation vs Neural Network Surrogate')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(t, omega_real, label='Real (scipy)', color='steelblue', lw=2)
    ax2.plot(t, omega_nn,  label='Neural Network', color='tomato', lw=2, linestyle='--')
    ax2.set_ylabel('Angular Velocity (rad/s)')
    ax2.set_xlabel('Time (s)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('nn_vs_simulation.png', dpi=150, bbox_inches='tight')
    plt.show()

    # Print error
    mae_theta = np.mean(np.abs(theta_real - theta_nn))
    print(f"Mean Absolute Error (theta): {mae_theta:.6f} rad")

# ─────────────────────────────
# CHECK MAIN
# ─────────────────────────────
if __name__ == "__main__":
    print("Generating data...")
    X, y = generate_data()

    print("Training network...")
    model, losses = train(X, y, epochs=1000)

    print("Comparing...")
    compare(model, theta0=0.3)