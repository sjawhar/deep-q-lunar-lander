Video report available [here](https://youtu.be/Gh3jBlLrEo4). Evaluation on OpenAI available [here](https://gym.openai.com/evaluations/eval_f2ZItfucTfe3m77eYgSew#reproducibility).

To launch locally, clone this repo then run this simple command in the project root folder:

```bash
docker-compose up -d
```
Then go to localhost:8888 in a web browser and open Project 2.ipynb. You can also `docker exec` into the running container and run `python runner.py`.

By default, the project uses the Keras implementation. Simple change `from learner_keras` to `from learner_numpy` to use the numpy implementation.

Using OpenAI Gym's monitor inside docker may or may not work for you based on your setup. On Windows, using VcXsrv with disabled access control works quite well.

Calling `solver.play(is_train=False, is_random=False)` tells the agent to play a single episode in the environment without random actions or any additional training. Use this with one of the persisted learners in the repo to observe fully-trained performance.
