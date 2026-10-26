import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

import utils.retrieval as retrieval


class RetrievalResourceTests(unittest.TestCase):
    def test_cross_encoder_initializes_once_for_concurrent_callers(self):
        instances = []

        def create_model(_model_name):
            time.sleep(0.02)
            instance = object()
            instances.append(instance)
            return instance

        with (
            patch.object(retrieval, "_cross_encoder", None),
            patch.object(retrieval, "CrossEncoder", side_effect=create_model),
            ThreadPoolExecutor(max_workers=6) as executor,
        ):
            models = list(
                executor.map(
                    lambda _index: retrieval._get_cross_encoder(), range(6)
                )
            )

        self.assertEqual(len(instances), 1)
        self.assertTrue(all(model is instances[0] for model in models))


if __name__ == "__main__":
    unittest.main()
