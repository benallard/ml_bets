
import click

import gym
from gym.wrappers.flatten_observation import FlattenObservation


from stable_baselines3 import PPO

import gym_euro
from dataset import EuroDataSet, bet_score, pred_to_score

@click.group()
def cli():
    pass

@cli.command()
def test():
    from stable_baselines3.common.env_checker import check_env
    env = gym.make('soccer-bet-v0')
    env = gym_euro.DeltaWrapper(env)
    check_env(env)

@cli.command()
@click.option("--env", "-E", "env_name", default="soccer-bet-v0")
@click.option("--load", "-l", "load_path")
@click.option("--timesteps", default=100000)
def train(env_name, load_path, timesteps):
    
    env = gym.make(env_name)

    env = gym_euro.DeltaWrapper(env)

    if load_path:
        model = PPO.load(load_path, env)
    else:
        model = PPO(
            'MlpPolicy',
            env,
            verbose=1,
        )


    model.learn(total_timesteps=timesteps)
    model.save("final_model")
    env.close()


@cli.command()
@click.option("--env", "-E", "env_name", default="soccer-bet-v0")
@click.option("--load", "-l", "load_path")
@click.option("--year")
def evaluate(env_name, load_path, year):
    env = gym.make(env_name)

    env = gym_euro.DeltaWrapper(env)
    model = PPO.load(load_path, env)

    data_set = EuroDataSet(year, False)
    points = 0
    for _in, out in data_set:
        pred, _ = model.predict(_in)
        #print (pred)
        expected = pred_to_score(pred)
        actual = pred_to_score(out)
        point = bet_score(expected, actual)
        print(f"Predicted {expected[0]:.0f}:{expected[1]:.0f} for {actual[0]:.0f}:{actual[1]:.0f} => +{point}")
        points += point
    print(f"Final points: {points}")

if __name__ == "__main__":
    cli()
