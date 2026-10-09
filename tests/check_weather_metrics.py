"""Physical-unit RMSE must preserve feature ordering and ignore zero-area poles."""
import torch
from experiments.evaluate_weather import evaluate


class Example:
    n_inputs = 2
    num_common_features = 2
    val_loss_fn = torch.nn.MSELoss()

    def __call__(self, fields):
        result = torch.zeros(1, 2, 3, 2)
        result[:, :, (0, 2)] = 1000  # Zero-area rows must not affect the score.
        return result


def main():
    inputs = torch.full((1, 1, 4, 3, 2), 3.)
    targets = torch.full((1, 1, 2, 3, 2), 3.)
    forcings = constants = torch.zeros(1, 1, 3, 2, 1)
    score, count, aggregate = evaluate(Example(), [(inputs, targets, forcings, constants)],
                            [1, 0], torch.tensor([2., 4.]), torch.tensor([0., 3., 0.]))
    assert count == 1 and score == [[6., 12.], [0., 0.]], score
    assert aggregate > 0
    print('PASS: field order, physical-unit scaling, area weights, persistence RMSE')


if __name__ == '__main__':
    main()
