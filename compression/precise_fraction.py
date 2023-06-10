import math


class PreciseFraction:
    def __init__(self, numerator, denominator):
        gcd = math.gcd(denominator, numerator)
        self.numerator = numerator//gcd
        self.denominator = denominator//gcd

    def __add__(self, other):
        denominator = math.lcm(self.denominator, other.denominator)
        numerator = self.numerator * (denominator // self.denominator) + other.numerator * (denominator // other.denominator)
        return PreciseFraction(numerator, denominator)

    def __sub__(self, other):
        return self + (other * -1)

    def __mul__(self, other):
        numerator = self.numerator * other.numerator
        denominator = self.denominator * other.denominator
        return PreciseFraction(numerator, denominator)

    def __eq__(self, other):
        return (self.numerator == other.numerator) & (self.denominator == other.denominator)

    def __lt__(self, other):
        denominator = math.lcm(self.denominator, other.denominator)
        return self.numerator * (denominator // self.denominator) < other.numerator * (denominator // other.denominator)

    def __le__(self, other):
        return (self < other) | (self == other)

    def __gt__(self, other):
        denominator = math.lcm(self.denominator, other.denominator)
        return self.numerator * (denominator // self.denominator) > other.numerator * (denominator // other.denominator)

    def __ge__(self, other):
        return (self > other) | (self == other)


    def __repr__(self):
        return f"{self.numerator}/{self.denominator}"

if __name__ == '__main__':
    a = PreciseFraction(1, 23476)
    b = PreciseFraction(1, 6)
    print(b * 3)