import torch


class ManualRankingModel(object):
    """ A model that always predict 1:0 for the team that is better ranked """

    def __call__(self, input):
        r_home, r_away = input[:2]
        # smaller ranking: better team
        return torch.tensor([1, 1, int(r_home < r_away), 0, int(r_away < r_home)])


class ManualOddModel(object):
    """ A model that always predict 1:0 for the team with the smaller odd"""

    def __call__(self, input):
        o_home = input[2]
        o_away = input[4]
        # smaller odd: better chance
        return torch.tensor([1, 1, int(o_home < o_away), 0, int(o_away < o_home)])


class ManualDrawModel(object):
    """ A model that always predict a draw 1:1"""

    def __call__(self, input):
        return torch.tensor([2, 0, 0, 1, 0])


class PredictorModel(object):
    DRAW_THRESHOLD = 0.5
    MAX_GOALS = 5
    DOMINATION_THRESHOLD = 20
    NONLINEARITY = 0.4

    """ https://github.com/fpoppinga/kicktipp-bot/blob/master/src/predictor/predictor.ts """

    def __call__(self, input):
        o_home = input[2].item()
        o_away = input[4].item()
        difference = abs(o_away - o_home)

        if difference < self.DRAW_THRESHOLD:
            return torch.tensor([2, 0, 0, 1, 0])

        totalGoals = min((difference / self.DOMINATION_THRESHOLD), 1) * self.MAX_GOALS
        if o_home > o_away:
            ratio = o_home / o_away
        else:
            ratio = o_away / o_home
        ratio = (ratio / (o_home + o_away)) ** self.NONLINEARITY

        winner = round(totalGoals * ratio)
        looser = round(totalGoals * (1 - ratio))

        if winner <= looser:
            winner += 1

        return torch.tensor([winner + looser, winner - looser, int(o_away > o_home), 0, int(o_home > o_away)])


def signof(number):
    return abs(number) / number
