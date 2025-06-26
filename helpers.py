import torch


def power_method(P, init_state, tol=1e-5, max_iter=200, return_intermed=False):
    """
    Computes the steady-state vector of a right stochastic matrix using the iterative power method.
    
    Parameters:
        P (torch.Tensor): A square right stochastic matrix of shape (n, n).
        init_state: (n, ) torch tensor representing the intial state, must sum up to 1
        tol (float): Tolerance for convergence.
        max_iter (int): Maximum number of iterations.
        return_intermed: returns intermediate outputs
    Returns:
        torch.Tensor: The steady-state vector (n, ).
    """
    x = init_state.detach().clone()
    x = x / x.sum()
    if return_intermed:
        intermeds = []
    for i in range(max_iter):
        if return_intermed:
            intermeds.append(x)
        x_new = torch.mv(P.T, x)          # Matrix-vector multiplication
        x_new = x_new / x_new.sum()       # Normalize
        # eigenvalue = x_new @ P.T @ x_new
        # print(eigenvalue)
        # Check convergence using L1 norm
        if torch.norm(x_new - x, p=1) < tol:
            if i > 0 and return_intermed:
                intermeds.append(x_new)
            break
        x = x_new
    if return_intermed:
        return x_new, torch.stack(intermeds)
    else:
        return x_new

def batch_power_method(P, init_state, max_iter=200):
    """
    Computes the steady-state vector of a right stochastic matrix using the iterative power method.
    
    Parameters:
        P (torch.Tensor): A square right stochastic matrix of shape (n, n).
        init_state: (n, ) torch tensor representing the intial state, must sum up to 1
        tol (float): Tolerance for convergence.
        max_iter (int): Maximum number of iterations.
        return_intermed: returns intermediate outputs
    Returns:
        torch.Tensor: The steady-state vector (n, ).
    """
    x = init_state.detach().clone()
    x = x / x.sum()
    x = x[None, ...]
    orig_shape = P.shape
    P_view = P.view(-1, orig_shape[-2], orig_shape[-1]).transpose(-1, -2)
    for i in range(max_iter):
        x_new = P_view @ x[..., None]  # Matrix-vector multiplication
        x_new = x_new[..., 0]
        x_new = x_new / x_new.sum(axis=-1, keepdims=True)  # Normalize
        x = x_new
    return x_new.view(orig_shape[:-1])


def prep_for_power_method(attention, alpha=0.99, personalization_vec=None):
    """
    Given a right stochastic matrix, computes an irreducible primitive Markov chain
    this gaurntees existance and uniqueness of a steady-state vector.
    Parameters:
        attention (torch.Tensor): A square right stochastic matrix of shape (..., n, n) (... can be any number of leading dims)
        alpha: the hyperparameter controlling bias, must have 0 < alpha < 1
                larger alpha: less bias
                smaller alpha: more emphasis on personalization vector
        personalization_vec: similar to PageRank. if None, will use uniform vector
    Returns:
        torch.Tensor: A square right stochastic matrix of shape (.., n, n) with a unique steady-state.
    """
    sequence_size = attention.shape[-1]
    if personalization_vec is None:
        e = torch.ones(sequence_size, dtype=attention.dtype).to(attention.device)
    else:
        e = personalization_vec.to(attention.device)
    E = e[:, None] @ e[None, :] / sequence_size
    T = alpha*(attention) + (1-alpha)*E
    return T


if __name__ == "__main__":
    # sanity checks for methods above
    # create a random batch of right sotchastic matrices
    batch_size = 5
    layers_size = 10
    spatial_dims = 64
    matrices = torch.rand(batch_size, layers_size, spatial_dims, spatial_dims)
    matrices = matrices / matrices.sum(dim=-1, keepdims=True)
    # prepare it for the power method, to avoid zero entries
    matrices = prep_for_power_method(matrices)
    # compute the first eigenvector using the eig method (slow)
    eigvals, eigvecs = torch.linalg.eig(matrices.transpose(-1, -2))
    eigvals = torch.abs(eigvals)
    values, indices = torch.topk(eigvals, 2, dim=-1)
    indices = indices.reshape(*indices.shape[:-1], 1, indices.shape[-1])
    expander = (-1,)*len(eigvecs.shape[:-2]) + (eigvecs.shape[-1],) + (-1,)
    indices = indices.expand(expander)
    eigvecs_largest = torch.gather(eigvecs, -1, indices)[..., 0]  # eigvecs[..., :, max(eigvalue)]
    ss = torch.abs(eigvecs_largest)
    computed_ss = (matrices.transpose(-1, -2) @ ss[..., None]).squeeze()
    assert torch.all(torch.isclose(computed_ss, ss))
    ss_sumnorm = ss / ss.sum(dim=-1, keepdims=True)
    # compute using the matrix power method (faster)
    mat = matrices[0, 0]
    as_exp = torch.linalg.matrix_power(mat, 30)
    ss1 = as_exp.mean(dim=-2) # deal with multiple steady states (naive: take first row)
    assert torch.all(torch.isclose(ss1, ss_sumnorm[0, 0]))
    # compute using the power method (fastest)
    init_state = torch.ones(mat.shape[-1], dtype=mat.dtype).to(mat.device) / mat.shape[-1]
    ss2 = power_method(mat, init_state)
    assert torch.all(torch.isclose(ss2, ss_sumnorm[0, 0]))
    # test the batched version also
    ss3 = batch_power_method(matrices, init_state)
    assert torch.all(torch.isclose(ss3, ss_sumnorm))

