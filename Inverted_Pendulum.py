import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

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

        #Defining a random Wind Disturbance
        u_wind = np.random.normal(0, 1.2)
        
        # Equation of Motion
        # Ml^2* domega = mg sin alpha L - D omega -F
        d_theta = omega
        inertia = self.m * self.L**2
        u_total = u_control + u_wind
        d_omega = (self.g / self.L) * np.sin(theta) - (self.D / inertia) * omega + (u_total / inertia)
        
        d_integral = error
        
        return [d_theta, d_omega, d_integral]

    def simulate(self, duration=10, theta0=0.2, omega0= 0):
        t_span = (0, duration)
        t_eval = np.linspace(0, duration, 1000)
        y0 = [theta0, omega0, 0.0]  # [Angle, Angular Velocity, Integral of Errors]
        
        sol = solve_ivp(self.dynamics, t_span, y0, t_eval=t_eval, method='RK45')
        return sol

def plot_results(sol):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    # Time Series Plot
    ax1.plot(sol.t, np.degrees(sol.y[0]), lw=2, label='Pendulum Angle')
    ax1.axhline(0, color='r', ls='--', alpha=0.5, label='Setpoint')
    ax1.set_ylabel('Angle (Degrees)')
    ax1.set_title('Inverted Pendulum PID Control Performance')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Phase Portrait
    ax2.plot(np.degrees(sol.y[0]), sol.y[1], color='purple', lw=1.5)
    ax2.set_xlabel('Angle (Degrees)')
    ax2.set_ylabel('Angular Velocity (rad/s)')
    ax2.set_title('Phase Space Trajectory (Convergence to Origin)')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('inverted_pendulum_control.png', dpi=150, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    result = InvertedPendulum().simulate(theta0=0.3)
    plot_results(result)