from pystrict import strict


class _Settings:
    pass


@strict
class MarineArctic(_Settings):
    color = 'cyan'


@strict
class MarineAverage(_Settings):
    color = 'blue'


@strict
class RuralContinental(_Settings):
    color = 'green'


@strict
class PollutedContinental(_Settings):
    color = 'red'
