setting = {'width': 8, 'height': 8, 'block_size': 20, 'speed': 20, 'layers': [67, 100, 67, 3],
           'vision mode': 4}
# set0
# [6, 15, 8, 3]=>max() binary
# [6, 15, 10, 6, 3]=>max() binary
# [7, 15, 10, 6, 3]=>max() binary
# [7, 25, 20, 15, 10, 6, 3]=>max() binary
# [7, 25, 20, 15,10,10, 10, 6, 3]=>max() binary 1000:30
# [7, 15, 30, 15, 3]=>max() binary 100:


# set1
# [19, 15, 8, 4]=>max(-) float

# set2
# [6, 15, 8, 3]=>max() float
# [6, 15, 10, 6, 3]=>max() float
# [6, 15, 10, 6, 3]=>max() float

# set3
# [10, 20, 20, 3]=>mean(214),max(42+) binary 2000:250
# [10, 20, 30, 20, 3]=>mean(214),max(42+) binary 2000:550
# [6, 20, 30, 20, 3]=>mean(110),max(25) binary
