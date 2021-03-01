import signatory
import torch


def get_insertion_matrix(signature, insertion_position, depth, channels):
    """This function creates the matrix corresponding to the insertion map, used in the optimization problem.

     Arguments:
        signature: torch.Tensor, shape (batch, channels + channels^2 + ... + channels^depth)
            Batch of signatures truncated at order depth: output of signatory.signature(path, depth), where path is a
            tensor of shape (batch, length, channels)
        
        insertion_position: int
            Insertion spot.

        depth: int
            Depth of the signature.
        
        channels: int
            The number of channels of the paths that were used to compute signature.

    Returns:
        A: torch.Tensor, shape (batch, channels, channels^(depth+1))
            Matrix A representing the linear insertion map.
    """

    batch = signature.shape[0]
    B = torch.cat(batch * [torch.eye(channels)])
    new_shape = [batch] + [channels] + [1] * (insertion_position-1) + [channels] + [1] * (depth + 1 -
                                                                                          insertion_position)
    repeat_points = [1, 1] + [channels] * (insertion_position-1) + [1] + [channels] * (depth + 1 - insertion_position)
    new_B = B.view(new_shape)
    new_B = new_B.repeat(repeat_points)

    last_signature_term = signatory.extract_signature_term(signature, channels, depth)
    last_signature_term = last_signature_term.view([batch] + [channels] * int(depth)).unsqueeze(insertion_position)
    repeat_points_sig = [1, channels] + [1] * (insertion_position - 1) + [channels] + [1] * (depth + 1 - insertion_position)
    sig_new_tensor = last_signature_term.unsqueeze(1).repeat(repeat_points_sig)

    A = (new_B * sig_new_tensor).flatten(start_dim=2)
    return A


def solve_optimization_problem(signature, insertion_position, depth, channels):
    """This function solves the optimization problem that allows to approximate the derivatives of the path.
        of the path.

    Arguments:
        signature : torch.Tensor, shape (batch, channels + channels^2 + ... + channels^depth)
            Batch of signatures truncated at order depth: output of signatory.signature(path, depth), where path is a
            tensor of shape (batch, length, channels)

        insertion_position : int
            Insertion spot.

        depth : int
            Depth of the signature.
        
        channels : int
            The number of channels of the paths that were used to compute signature.

    Returns:
        x_optimal: torch.Tensor, shape (batch, channels)
            Solution of the optimization problem, approximation of the derivatives of the paths on the interval
            indexed by insertion_position.
    """

    A_matrix = get_insertion_matrix(signature[:, :-channels ** depth], insertion_position, depth - 1, channels)
    b_vector = depth * signatory.extract_signature_term(signature, channels, depth)

    x_optimal = torch.matmul(A_matrix, b_vector.unsqueeze(-1)).squeeze(-1)

    sign_1 = signatory.extract_signature_term(signature, channels, depth - 1)

    return x_optimal / (torch.norm(sign_1, dim=1).unsqueeze(-1) ** 2)


def invert_signature(signature, depth, channels, initial_position=None):
    """Reconstruct the path from its signature

    Arguments:
        signature: torch.Tensor, shape (batch, channels + channels^2 + ... + channels^depth)
            Batch of signatures truncated at order depth: output of signatory.signature(path, depth), where path is a
            tensor of shape (batch, length, channels)

        depth: int
            Depth of the signature.
        
        channels: int
            The number of channels of the paths that were used to compute signature.

        initial_position: optional, torch.Tensor, shape (batch, channels)
            Initial point of the paths. If None, the reconstructed paths are set to begin at zero.

    Returns:
        path: torch.Tensor, shape (batch, depth+1, channels)
            Batch of inverted paths.
    """
    if signatory.signature_channels(channels, depth) != signature.shape[1]:
        raise ValueError("channels and depth do not correspond to signature shape.")

    batch = signature.shape[0]
    path_derivatives = torch.zeros((batch, depth, channels))
    path = torch.zeros((batch, depth+1, channels))
    
    if initial_position is not None:
        path[:, 0, :] = initial_position

    if depth == 1:
        path[:, 1, :] = path[:, 0, :] + signature
    else:
        for insertion_position in torch.arange(1, depth + 1):
            x_optimal = solve_optimization_problem(signature, insertion_position, depth, channels)
            path_derivatives[:, insertion_position-1, :] = x_optimal
            path[:, insertion_position, :] = (path[:, insertion_position - 1, :]
                                              + path_derivatives[:, insertion_position - 1, :] * (1 / depth))

    return path


if __name__ == '__main__':
    depth_test = 1
    channels_test = 2
    batch_test = 10
    
    test_path = torch.rand((batch_test, 100, channels_test))
    signature_test = signatory.signature(test_path, depth_test)
    invert_signature(signature_test, depth_test, channels_test).shape
