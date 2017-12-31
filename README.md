## Usage

To launch locally, clone this repo then simply run `docker-compose up -d` in the project root folder. Then go to localhost:8888 in a web browser and open runner.ipynb. You can also `docker exec` into the running container and run `python runner.py`.

Using OpenAI Gym's monitor inside docker may or may not work for you based on your setup. On Windows, VcXsrv works quite well.

Calling `solver.play(is_train=False, is_random=False)` tells the agent to play a single episode in the environment without random actions or any additional training. Use this with one of the persisted learners in the repo to observe fully-trained performance.
