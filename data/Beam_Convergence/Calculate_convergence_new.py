import numpy as np
import matplotlib.pyplot as plt

# Material Properties
E = 1000
Iz = 1/12
Iy =1/12
J = 0
L = 10

# Load Case
q_y = -10

def exact_deflection(x):
    return q_y * (x**4 - 4 * L * x**3 + 6 * L**2 * x**2) / (24 * E * Iz)

# Shape function for beam bending deflection in v and \theta_z
def N_matrix(x, Mesh_density):
    N = np.zeros(4)
    Le = L / Mesh_density  # Element length
    xi = 2 * x / Le - 1  # Parent coordinate xi
    N[0] = (xi**3 - 3 * xi + 2) / 4
    N[1] = Le * (xi**3 - xi**2 - xi + 1) / 8
    N[2] = (-xi**3 + 3 * xi + 2) / 4
    N[3] = Le * (xi**3 + xi**2 - xi - 1) / 8
    return N

# Obtain displacement data for different mesh densities
d1 = [0, 0, -150, -20]
d2 = [0, 0, -53.2, -17.53, -53.2, -17.53, -150.3, -20.06]
d5 = [0, 0, -10.48, -9.76, -10.48, -9.76, -36.48, -15.68, -36.48, -15.68, -71.28, -18.72, -71.28, -18.72, -110.08, -19.84, -110.08, -19.84, -150, -20]
d10 = [0, 0, -2.805, -5.42, -2.805, -5.42, -10.48, -9.76, -10.48, -9.76, -22.005, -13.14, -22.005, -13.14, -36.48, -15.68, -36.48, -15.68, -53.125, -17.5, -53.125, -17.5, -71.28, -18.72, -71.28, -18.72, -90.405, -19.46, -90.405, -19.46, -110.08, -19.84, -110.08, -19.84, -130.005, -19.98, -130.005, -19.98, -150, -20]
d20 = [0, 0, -0.725313, -2.8525, -0.725313, -2.8525, -2.805, -5.42, -2.805, -5.42, -6.10031, -7.7175, -6.10031, -7.7175, -10.48, -9.76, -10.48, -9.76, -15.8203, -11.5625, -15.8203, -11.5625, -22.005, -13.14, -22.005, -13.14, -28.9253, -14.5075, -28.9253, -14.5075, -36.48, -15.68, -36.48, -15.68, -44.5753, -16.6725, -44.5753, -16.6725, -53.125, -17.5, -53.125, -17.5, -62.0503, -18.1775, -62.0503, -18.1775, -71.28, -18.72, -71.28, -18.72, -80.7503, -19.1425, -80.7503, -19.1425, -90.405, -19.46, -90.405, -19.46, -100.195, -19.6875, -100.195, -19.6875, -110.08, -19.84, -110.08, -19.84, -120.025, -19.9325, -120.025, -19.9325, -130.005, -19.98, -130.005, -19.98, -140, -19.9975, -140, -19.9975, -150, -20]

d = [d1, d2, d5, d10, d20]

# Calculate x-coordinates for Gaussian points in each element
def calculate_moment_x_coords(Mesh_density, num_elements):
    x_coords = []
    delta_x = 10 / Mesh_density
    for I in range(1, num_elements + 1):
        x1 = (delta_x * (I )) + (delta_x / 2) - (delta_x / 2) * np.sqrt(3) / 3
        x2 = (delta_x * (I )) + (delta_x / 2) + (delta_x / 2) * np.sqrt(3) / 3
        x_coords.extend([x1, x2])
    return np.array(x_coords)

# Placeholder function for exact deflection
def exact_deflection(x):
    return -0.005 * (x**4 - 40 * x**3 + 600 * x**2)

# FEM solutions by shape functions
def v_FEM(x, d, Mesh_density):
    v = np.zeros(Mesh_density)  # Create an array for the deflection at each point x
    for idx, xi in enumerate(x):
        element_idx = min(int(xi // (L / Mesh_density)), Mesh_density - 1)  # Ensure element_idx is within bounds
        local_x = xi - (element_idx * (L / Mesh_density))
        N = N_matrix(local_x, Mesh_density)
        element_start_idx = element_idx * 4
        v[idx] = (N[0] * d[element_start_idx] +
                  N[1] * d[element_start_idx + 1] +
                  N[2] * d[element_start_idx + 2] +
                  N[3] * d[element_start_idx + 3])
    return v

# Calculate L2 error of deflection (v) for different mesh densities
def calculate_deflection_error(data_deflection, Mesh_density):
    x_coords = calculate_moment_x_coords(Mesh_density, Mesh_density)
    gp1 = x_coords[::2]
    gp2 = x_coords[1::2]
    
    # Gaussian quadrature weights for two-point rule
    weights = [1, 1]  # For two-point Gaussian quadrature
    
    L2_error = 0
    for i in range(Mesh_density):
        v_fem_gp1 = v_FEM([gp1[i]], data_deflection, Mesh_density)[0]
        v_fem_gp2 = v_FEM([gp2[i]], data_deflection, Mesh_density)[0]
        v_exact_gp1 = exact_deflection(gp1[i])
        v_exact_gp2 = exact_deflection(gp2[i])
        
        L2_error += weights[0] * (v_fem_gp1 - v_exact_gp1)**2 + weights[1] * (v_fem_gp2 - v_exact_gp2)**2
    
    L2_error = np.sqrt(L2_error)  # Scale by the interval length divided by 2 for Gaussian quadrature
    
    return L2_error

# Calculate errors and element lengths
errors = []
element_lengths = []
for i, data_deflection in enumerate(d):
    Mesh_density = len(data_deflection) // 4
    error = calculate_deflection_error(data_deflection, Mesh_density)
    errors.append(error)
    element_lengths.append(L / Mesh_density)
    print(f'L2_error for deflection (v) of Meshdensity={Mesh_density}: {error}')

# Plot log(L2_error) vs log(Le)
log_errors = np.log(errors)
log_element_lengths = np.log(element_lengths)

plt.figure(figsize=(10, 6))
plt.plot(log_element_lengths, log_errors, marker='o', linestyle='-', label='Log-Log Plot')
plt.xlabel('log(Element Length)')
plt.ylabel('log(L2 Error)')
plt.title('Log-Log Plot of L2 Error vs Element Length')
plt.legend()
plt.grid(True)
plt.show()

# Calculate the slope of the log-log plot
slope, intercept = np.polyfit(log_element_lengths, log_errors, 1)
print(f'Slope of the log-log plot: {slope}')


