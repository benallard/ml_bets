import torch

class ManualRankingModel(object):
    """ API compatible with a torch Module """
    def __call__(self, input):
        r_home, r_away = input[:2]
        # smaller ranking: better team
        return torch.tensor([1, 1, int(r_home < r_away), 0, int(r_away < r_home)])

class ManualOddModel(object):
    def __call__(self, input):
        o_home = input[2]
        o_away = input[4]
        # smaller odd: better chance
        return torch.tensor([1, 1, int(o_home < o_away), 0, int(o_away < o_home)])

class ManualDrawModel(object):
    def __call__(self, input):
        return torch.tensor([2, 0, 0, 1, 0])

def signof(number):
    return abs(number) / number