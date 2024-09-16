import time

import requests

if __name__ == "__main__":
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

    start_time = time.time()

    # Send inference request to model and ensure it returned a 200 status code
    response = requests.post(
        url="http://localhost:8000/v2/models/topic_summarization/infer",
        json=json_input,
        timeout=600,
    )

    end_time = time.time()

    print(response.text)
    print("\n")
    print(f"Processing time: {end_time - start_time}")
