
import click

import gym
from gym.wrappers.flatten_observation import FlattenObservation


from stable_baselines3 import PPO

import gym_euro

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
        # Note that you may override use_masking value loaded from previous model
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


if __name__ == "__main__":
    cli()
