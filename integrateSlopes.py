import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import lsqr

def reconstruct_surface_least_squares(sx_measured, sy_measured, dx=1.0, dy=1.0, fixed_point=(0, 0, 0)):
    """
    Reconstructs a 2D surface from its discrete x and y slopes using a least-squares approach.

    Args:
        sx_measured (np.ndarray): 2D array of measured slopes in the x-direction.
                                  Expected shape: (rows, cols) if using forward diff.
        sy_measured (np.ndarray): 2D array of measured slopes in the y-direction.
                                  Expected shape: (rows, cols) if using forward diff.
        dx (float): Grid spacing in the x-direction.
        dy (float): Grid spacing in the y-direction.
        fixed_point (tuple): (row, col, value) for a known point on the surface
                             to establish the absolute height. If None, the average
                             height will be set to 0.

    Returns:
        np.ndarray: Reconstructed 2D surface.
    """
    # Infer target surface dimensions from slope arrays
    # If sx_measured is (R, C-1), then surface has C columns
    # If sy_measured is (R-1, C), then surface has R rows
    # We take the dimensions from sx_measured for cols and sy_measured for rows,
    # then adjust based on the expected difference array sizes.
    
    # Assuming sx_measured is (rows, cols-1) and sy_measured is (rows-1, cols)
    # The target surface will be (rows_surface, cols_surface)
    rows_surface = sy_measured.shape[0]
    cols_surface = sx_measured.shape[1]
    


    # Total number of unknown surface points
    N = rows_surface * cols_surface

    # Number of equations for Sx + Sy + 1 for fixed point/mean constraint
    num_equations_sx = sx_measured.size # rows_surface * cols_surface
    num_equations_sy = sy_measured.size # rows_surface * cols_surface
    num_equations = num_equations_sx + num_equations_sy

    # Initialize sparse matrix A and vector b
    # A will be (num_equations + 1) x N (for fixed point constraint)
    A = lil_matrix((num_equations + 1, N), dtype=float)
    b = np.zeros(num_equations + 1, dtype=float)

    eq_idx = 0

    # Equations for Sx (Z(i, j+1) - Z(i, j) = Sx(i, j) * dx)
    # Loop over the actual dimensions of sx_measured
    for i in range(rows_surface): # Iterates through rows of the surface
        for j in range(cols_surface - 1): # Iterates through columns of sx_measured
            # Index for Z(i, j) in the flattened array
            idx_curr = i * cols_surface + j
            # Index for Z(i, j+1)
            idx_next = i * cols_surface + (j + 1)

            A[eq_idx, idx_next] = 1.0 / dx
            A[eq_idx, idx_curr] = -1.0 / dx
            b[eq_idx] = sx_measured[i, j]
            eq_idx += 1

    # Equations for Sy (Z(i+1, j) - Z(i, j) = Sy(i, j) * dy)
    # Loop over the actual dimensions of sy_measured
    for i in range(rows_surface - 1): # Iterates through rows of sy_measured
        for j in range(cols_surface): # Iterates through columns of the surface
            # Index for Z(i, j) in the flattened array
            idx_curr = i * cols_surface + j
            # Index for Z(i+1, j)
            idx_next = (i + 1) * cols_surface + j

            A[eq_idx, idx_next] = 1.0 / dy
            A[eq_idx, idx_curr] = -1.0 / dy
            b[eq_idx] = sy_measured[i, j]
            eq_idx += 1

    # Add a constraint for the absolute height (fixed point or mean)
    if fixed_point is not None:
        fp_row, fp_col, fp_val = fixed_point
        # Convert fixed point (row, col) to flattened index
        fp_idx = fp_row * cols_surface + fp_col
        A[eq_idx, fp_idx] = 1.0
        b[eq_idx] = fp_val
    else:
        # Constraint to set the mean height of the surface to 0
        for k in range(N):
            A[eq_idx, k] = 1.0 / N
        b[eq_idx] = 0.0

    # Convert to CSR format for efficient matrix-vector products
    A = A.tocsr()

    # Solve the linear system A @ z_flat = b using least squares
    # lsqr is good for sparse matrices
    z_flat, istop, itn, r1norm = lsqr(A, b)[:4]

    if istop != 1: # istop = 1 means x is approximate solution to Ax = b
        print(f"Warning: lsqr did not converge optimally. istop = {istop}")

    # Reshape the flattened solution back to the 2D grid
    reconstructed_surface = z_flat.reshape(rows_surface, cols_surface)

    return reconstructed_surface

# # --- Example Usage ---
# if __name__ == "__main__":
#     # 1. Generate a synthetic true surface
#     rows, cols = 50, 60
#     x = np.linspace(0, 10, cols)
#     y = np.linspace(0, 8, rows)
#     X, Y = np.meshgrid(x, y)

#     # Example surface: a paraboloid with some tilt
#     true_surface = 0.1 * X**2 + 0.05 * Y**2 + 0.5 * X + 0.2 * Y + 5.0

#     # 2. Calculate its discrete slopes (using numpy.gradient for ground truth slopes)
#     # np.gradient returns dy, dx (vertical, horizontal)
#     # We need to extract the correct parts for sx_true_truncated and sy_true_truncated
#     # as our solver expects slopes representing forward differences:
#     # Sx[i,j] is (Z[i,j+1] - Z[i,j])/dx
#     # Sy[i,j] is (Z[i+1,j] - Z[i,j])/dy

#     # Let's compute these directly from the true_surface to match our solver's expectation
#     dx_spacing = x[1] - x[0]
#     dy_spacing = y[1] - y[0]

#     # Sx_true: difference along columns
#     # Shape will be (rows, cols-1)
#     sx_true = (true_surface[:, 1:] - true_surface[:, :-1]) / dx_spacing

#     # Sy_true: difference along rows
#     # Shape will be (rows-1, cols)
#     sy_true = (true_surface[1:, :] - true_surface[:-1, :]) / dy_spacing


#     # 3. Add some noise to the slopes
#     noise_level = 0.1
#     sx_noisy = sx_true + noise_level * np.random.randn(*sx_true.shape)
#     sy_noisy = sy_true + noise_level * np.random.randn(*sy_true.shape)

#     # 4. Reconstruct the surface
#     # Choose a fixed point for absolute height. Let's fix the value at (0,0) to be the true value.
#     fixed_point_coord = (0, 0, true_surface[0, 0])
#     # Or, to just set the mean to zero, pass fixed_point=None

#     reconstructed_surface = reconstruct_surface_least_squares(
#         sx_noisy, sy_noisy, dx=dx_spacing, dy=dy_spacing, fixed_point=fixed_point_coord
#     )

#     # --- Visualization ---
#     fig = plt.figure(figsize=(15, 6))

#     # Plot True Surface
#     ax1 = fig.add_subplot(121, projection='3d')
#     ax1.plot_surface(X, Y, true_surface, cmap='viridis', alpha=0.8)
#     ax1.set_title('True Surface')
#     ax1.set_xlabel('X')
#     ax1.set_ylabel('Y')
#     ax1.set_zlabel('Z')

#     # Plot Reconstructed Surface
#     ax2 = fig.add_subplot(122, projection='3d')
#     ax2.plot_surface(X, Y, reconstructed_surface, cmap='viridis', alpha=0.8)
#     ax2.set_title('Reconstructed Surface (Least Squares)')
#     ax2.set_xlabel('X')
#     ax2.set_ylabel('Y')
#     ax2.set_zlabel('Z')

#     plt.tight_layout()
#     plt.show()

#     # Calculate and print reconstruction error
#     rmse = np.sqrt(np.mean((true_surface - reconstructed_surface)**2))
#     print(f"RMSE between true and reconstructed surface: {rmse:.4f}")

#     # You can also visualize the difference
#     plt.figure(figsize=(8, 6))
#     plt.imshow(true_surface - reconstructed_surface, cmap='coolwarm', origin='lower',
#                extent=[x.min(), x.max(), y.min(), y.max()])
#     plt.colorbar(label='Difference (True - Reconstructed)')
#     plt.title('Difference between True and Reconstructed Surface')
#     plt.xlabel('X')
#     plt.ylabel('Y')
#     plt.show()