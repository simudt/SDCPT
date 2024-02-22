import torch

from sdcpt.distance.bray_curtis import BrayCurtisDistance


def test_BrayCurtisDistance():
    array1 = [1, 3, 27]
    array2 = [3, 6, 8]

    bray_curtis_calculator = BrayCurtisDistance(array1, array2)

    distance = bray_curtis_calculator.compute_distance()

    print("Bray-Curtis Distance:", distance)


test_BrayCurtisDistance()
