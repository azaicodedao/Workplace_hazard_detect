import unittest

import torch

from app.models.model_loader import add_missing_videomae_key_biases


class VideoMAECheckpointCompatibilityTests(unittest.TestCase):
    def test_adds_only_missing_key_bias_as_zero(self):
        key_bias = "videomae.encoder.layer.0.attention.attention.key.bias"
        query_bias = "videomae.encoder.layer.0.attention.attention.query.bias"
        state_dict = {query_bias: torch.ones(3)}
        model_state = {
            query_bias: torch.full((3,), 9.0),
            key_bias: torch.full((3,), 7.0),
        }

        result = add_missing_videomae_key_biases(state_dict, model_state)

        self.assertTrue(torch.equal(result[query_bias], torch.ones(3)))
        self.assertTrue(torch.equal(result[key_bias], torch.zeros(3)))

    def test_does_not_overwrite_existing_key_bias(self):
        key_bias = "videomae.encoder.layer.0.attention.attention.key.bias"
        existing = torch.full((3,), 2.0)
        state_dict = {key_bias: existing}

        result = add_missing_videomae_key_biases(
            state_dict,
            {key_bias: torch.full((3,), 7.0)},
        )

        self.assertIs(result[key_bias], existing)


if __name__ == "__main__":
    unittest.main()
