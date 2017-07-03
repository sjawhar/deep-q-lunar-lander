To launch locally, clone this repo then run this simple command in the project root folder:

```bash
docker-compose up -d
```
Then go to localhost:8888 in a web browser and open Project 2.ipynb. You can also `docker exec` into the running container and run `python runner.py`.

By default, the project uses the Keras implementation. Simple change `from learner_keras` to `from learner_numpy` to use the numpy implementation.

Using OpenAI Gym's monitor inside docker may or may not work for you based on your setup. On Windows, using VcXsrv with disabled access control works quite well.
