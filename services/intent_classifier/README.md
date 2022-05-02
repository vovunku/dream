# DNNC as service
## Overview

Service to provide intent classifier functionality via http.

## Requirements

You need docker installed. Also you have to prepare python environment by running

```bash
pip install requirements.txt
```

Prepare approx. 2GB of RAM. If you want to use GPU then you will need cuda installed with approx 2GB of GPU

## Running

```sh
docker build . --rm=true -t dff_service --build-arg SERVICE_PORT=1234 --build-arg TRAIN_DATA_PATH=/home/vovun/dream/services/intent_classifier/data/example.json
sudo docker run --name dff_s_container --rm -p1234:1234 -t dff_service
```

To stop(bad method):

```sh
docker container ls
docker container rm 14fdc24098b5 --force
```

## API
* `/respond` - expects dict ```{"text": "user query"}```. Then respons will be in json format: ```{"predicted": intent_prediction}```. Intent prediction contains ```[predicted_intent, similarity, closest_train_example]```.
## Usage

Example usage
```python
import requests
res = requests.post('http://0.0.0.0:1234/respond', json={"text": "Bye pal"})
print(res.json())
```

```bash
curl -X POST http://0.0.0.0:1234/respond -H 'Content-Type: application/json' -d '{"text": "Bye pal"}'
```

Expected result:
```js
{"predicted":["bye",0.8378442525863647,"Goodbye friend!"]}
```

## Training data

We use json format for training data as following:

```json
{
    "greet": [
        "Howdy friend!",
        "Hello dear",
        "Oh hi pal!",
        "Hello everyone",
        "Bonjour mon ami",
        "Nice to see you again",
        "Good evening!"
    ],
    "bye": [
        "Bye bye sweetie!",
        "See you later",
        "Goodbye friend!",
        "See you soon!",
        "Alright then"
    ]
}
```

## Porperties

~2 ms for query. Depends on training data.
