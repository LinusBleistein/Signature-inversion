from insertion import get_insertion_matrix, solve_optimization_problem, invert_signature
import torch
import signatory


def test_insertion_matrix_shape():
    """ Tests that the insertion matrix obtained is of the right shape"""
    for batch_size in (1, 2, 5):
        for input_stream in (2, 3, 10):
            for input_channels in (1, 2, 6):
                path = torch.rand((batch_size, input_stream, input_channels))
                for depth in (1, 2, 4, 6):
                    signature = signatory.signature(path, depth)
                    correct_shape = (batch_size, input_channels, input_channels ** (depth+1))
                    for insertion_position in range(1, depth + 1):
                        insertion_matrix = get_insertion_matrix(signature, insertion_position, depth, input_channels)
                    assert insertion_matrix.shape == correct_shape


def test_insertion_matrix_structure():
    """Tests that the insertion matrix has identical singular values, all equal to the norm of the signature of order
    depth"""
    batch_size = 1
    for input_stream in (2, 3, 10):
        for input_channels in (1, 2, 6):
            path = torch.rand((batch_size, input_stream, input_channels))
            for depth in (1, 2, 4, 6):
                signature = signatory.signature(path, depth)
                norm_sign = torch.norm(signatory.extract_signature_term(signature, input_channels, depth)[0]) ** 2
                for insertion_position in range(1, depth + 1):
                    insertion_matrix = get_insertion_matrix(signature, insertion_position, depth, input_channels)[0]
                    diagonal_matrix = torch.matmul(insertion_matrix, insertion_matrix.T)
                    assert diagonal_matrix.shape == (input_channels, input_channels)
                    assert torch.allclose(diagonal_matrix[0, 0], norm_sign, atol=1e-05)
                    if input_channels > 1:
                        assert diagonal_matrix[0, input_channels-1] == 0.
                        assert torch.allclose(diagonal_matrix[input_channels-1, input_channels-1],
                                              diagonal_matrix[0, 0])


def test_inverted_path_shape():
    """Tests that the inverted path is of the right shape"""
    for batch_size in (1, 2, 5):
        for input_stream in (2, 3, 10):
            for input_channels in (1, 2, 6):
                path = torch.rand((batch_size, input_stream, input_channels))
                for depth in (2, 4, 6):
                    signature = signatory.signature(path, depth)
                    inverted_path = invert_signature(signature, depth, input_channels, initial_position=path[:, 0, :])
                    assert inverted_path.shape == (batch_size, depth + 1, input_channels)


def test_initial_position_zero():
    """Tests that the inverted path initial position is the right one."""
    batch_size = 10
    input_stream = 10
    input_channels = 3
    path = torch.rand((batch_size, input_stream, input_channels))
    for depth in (2, 4, 6):
        signature = signatory.signature(path, depth)
        inverted_path = invert_signature(signature, depth, input_channels)
        assert torch.equal(inverted_path[:, 0, :], torch.zeros(batch_size, input_channels))
        initial_position = torch.rand((batch_size, input_channels))
        inverted_path = invert_signature(signature, depth, input_channels, initial_position=initial_position)
        assert torch.equal(inverted_path[:, 0, :], initial_position)

