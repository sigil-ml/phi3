"""
Test each known model in Nvidia Triton to ensure each was loaded properly & returns correct output.
"""

# import pickle
import unittest

# import warnings
# from pathlib import Path
import requests

# import torch

# import torch.nn.functional as F
# from sentence_transformers import SentenceTransformer


class TestTriton(unittest.TestCase):
    """Test models loaded by Triton for correct output"""

    # @classmethod
    # def setUpClass(cls):
    #     cls.base_url = "http://localhost:8000/v2/models"
    #     cls.device = "cuda" if torch.cuda.is_available() else "cpu"

    #     data_path = Path(__file__).absolute().parent / "data/test_data.pkl"
    #     with warnings.catch_warnings():
    #         warnings.simplefilter("ignore", category=FutureWarning)
    #         with open(data_path, "rb") as file:
    #             cls.test_data = pickle.load(file)

    #             model = SentenceTransformer(
    #                 "sentence-transformers/all-mpnet-base-v2", device=cls.device
    #             )

    #     document_embeddings = model.encode(cls.test_data["haiti_data"]["documents"])
    #     cls.target_document_embeddings = torch.tensor(
    #         document_embeddings, device=cls.device
    #     )

    def test_topic_summarization(self) -> None:
        """
        Test if the topic_summarization model can be used for inference & produces reasonable summaries
        """

        model_name = "topic_summarization"

        try:
            # Test that the model was loaded and available for inference
            response = requests.get(
                url=f"http://localhost:8000/v2/models/{model_name}/ready", timeout=60
            )
            self.assertEqual(
                response.status_code, 200, f"{model_name} not ready for inferencing"
            )

            data = [
                "This is a story about a dog that chased a mailman down the street.",
                "This is a story about a mailman that got chased by a dog while he was delivering mail.",
                "This is a story about a neighbor who watched a mailman get chased down the street by his neighbors dog!",
            ]

            # Create json data for the request
            json_input = {
                "inputs": [
                    {
                        "name": "documents",
                        "datatype": "BYTES",
                        "shape": [len(data)],
                        "data": data,
                    }
                ]
            }

            # Send inference request to model and ensure it returned a 200 status code
            response = requests.post(
                url=f"http://localhost:8000/v2/models/{model_name}/infer",
                json=json_input,
                timeout=60,
            )
            self.assertEqual(
                response.status_code,
                200,
                f"Received non 200 status code from {model_name} during inferencing",
            )

            # Parse results from model (should just be 'embeddings')
            # results = {
            #     i["name"]: torch.tensor(i["data"]).reshape(i["shape"]).to(self.device)
            #     for i in response.json()["outputs"]
            # }

            print(response)

        except requests.exceptions.ConnectionError:
            self.fail(f"{model_name} ConnectionError to Triton")


if __name__ == "__main__":
    unittest.main()
